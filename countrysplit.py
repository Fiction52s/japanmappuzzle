#!/usr/bin/env python3
# countrysplit.py — thin polyline cutter with undo/redo, pan/zoom,
#                   proper piece separation, and groove application

import os
import argparse
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Button
import shapely.geometry as geom
import trimesh


# ---------------- utilities ----------------

def robust_concat(obj):
    """Turn a trimesh boolean result (Trimesh | Scene | list) into a Trimesh or None."""
    if obj is None:
        return None
    if isinstance(obj, trimesh.Trimesh):
        return obj
    if isinstance(obj, trimesh.Scene):
        geoms = [g for g in obj.dump().geometry.values() if isinstance(g, trimesh.Trimesh)]
        if not geoms:
            return None
        return trimesh.util.concatenate(geoms)
    if isinstance(obj, (list, tuple)):
        geoms = []
        for x in obj:
            t = robust_concat(x)
            if isinstance(t, trimesh.Trimesh):
                geoms.append(t)
        if not geoms:
            return None
        return trimesh.util.concatenate(geoms)
    return None


def split_components(mesh: trimesh.Trimesh):
    """Split a mesh into components and drop tiny crumbs."""
    comps = mesh.split(only_watertight=False)
    if not comps:
        return []

    # keep components that are a sensible fraction of the largest
    areas = [float(c.area) for c in comps]
    lead = max(areas) if areas else 0.0
    kept = [c for c in comps if float(c.area) >= max(1e-6 * lead, 1e-3) and c.faces.shape[0] > 10]
    if not kept:
        kept = comps
    return kept


# ---------------- cutting ----------------

def make_wall_from_polyline(polyline, bounds, height_scale=4.0, wall_thickness=0.3):
    """
    Build a very thin extruded wall mesh along the polyline in XY.
    Thin kerf (default 0.3mm) so pieces fit neatly.
    """
    (minx, miny, minz), (maxx, maxy, maxz) = bounds
    span_z = (maxz - minz) * height_scale

    line = geom.LineString(polyline)
    if line.length == 0:
        raise ValueError("Cut line too short")

    # flat caps & miter joins keep corners crisp
    fp = line.buffer(wall_thickness, cap_style=2, join_style=2)
    wall = trimesh.creation.extrude_polygon(fp, span_z)
    wall.apply_translation([0, 0, minz - span_z / 2.0])
    return wall


def cut_mesh_with_polyline(mesh: trimesh.Trimesh, polyline, kerf=0.3):
    """
    Do a *single* boolean difference (mesh - thin wall), then split into
    disconnected components. Returns the separated pieces.
    """
    wall = make_wall_from_polyline(polyline, mesh.bounds, wall_thickness=kerf)

    # one difference is enough; splitting yields the two pieces
    try:
        diff = trimesh.boolean.difference([mesh, wall], engine="scad")
    except BaseException:
        diff = trimesh.boolean.difference([mesh, wall])

    diff = robust_concat(diff)
    if diff is None or diff.faces.shape[0] == 0:
        # fallback: keep original mesh if the boolean failed
        return [mesh.copy()]

    diff.remove_degenerate_faces()
    diff.remove_unreferenced_vertices()
    diff.fix_normals()

    return split_components(diff)


# ---------------- grooves ----------------

def maybe_flip_grooves_z(g: trimesh.Trimesh, about_z: float):
    """
    Mirror grooves along Z about the plane z=about_z.
    (Rotate 180° in Z-scale so grooves affect the opposite face.)
    """
    T = np.eye(4)
    S = np.eye(4)
    T[2, 3] = -about_z
    S[2, 2] = -1.0
    Ti = np.eye(4)
    Ti[2, 3] = about_z
    g.apply_transform(Ti @ S @ T)


def add_grooves_to_piece(piece: trimesh.Trimesh,
                         grooves_mesh: trimesh.Trimesh,
                         flip_grooves_z: bool):
    """
    Subtract the (pre-aligned) grooves mesh from this piece.
    Optionally mirror grooves along Z to hit the opposite face.
    """
    g = grooves_mesh.copy()
    if flip_grooves_z:
        # mirror grooves about the plate mid-Z (use grooves' own center as proxy)
        gz0, gz1 = g.bounds[0][2], g.bounds[1][2]
        maybe_flip_grooves_z(g, about_z=0.5 * (gz0 + gz1))

    try:
        out = piece.difference(g, engine="scad")
    except BaseException:
        out = piece.difference(g)

    out = robust_concat(out)
    if out is None:
        # If subtraction fails, return original piece
        return piece.copy()

    out.remove_degenerate_faces()
    out.remove_unreferenced_vertices()
    out.fix_normals()
    return out


# ---------------- pan/zoom ----------------

def connect_default_panzoom(fig, ax):
    state = {"drag": False, "last": None}

    def on_scroll(event):
        if event.inaxes != ax:
            return
        scale = 1.5 if event.button == 'up' else 1/1.5
        xlim = ax.get_xlim(); ylim = ax.get_ylim()
        x, y = event.xdata, event.ydata

        ax.set_xlim([x - (x - xlim[0]) / scale,
                     x + (xlim[1] - x) / scale])
        ax.set_ylim([y - (y - ylim[0]) / scale,
                     y + (ylim[1] - y) / scale])
        fig.canvas.draw_idle()

    def on_press(e):
        if e.button == 2 and e.inaxes == ax:
            state["drag"] = True
            state["last"] = e

    def on_release(e):
        if e.button == 2:
            state["drag"] = False
            state["last"] = None

    def on_motion(e):
        if not state["drag"] or state["last"] is None or e.inaxes != ax:
            return
        dx = e.xdata - state["last"].xdata
        dy = e.ydata - state["last"].ydata
        xlim = ax.get_xlim(); ylim = ax.get_ylim()
        ax.set_xlim(xlim[0] - dx, xlim[1] - dx)
        ax.set_ylim(ylim[0] - dy, ylim[1] - dy)
        state["last"] = e
        fig.canvas.draw_idle()

    fig.canvas.mpl_connect("scroll_event", on_scroll)
    fig.canvas.mpl_connect("button_press_event", on_press)
    fig.canvas.mpl_connect("button_release_event", on_release)
    fig.canvas.mpl_connect("motion_notify_event", on_motion)


# ---------------- GUI ----------------

class SplitGUI:
    def __init__(self, plate_mesh, grooves_mesh, preview_image_path, outdir,
                 kerf, flip_grooves_z):
        self.mesh = plate_mesh
        self.grooves = grooves_mesh
        self.kerf = kerf
        self.flip_grooves_z = flip_grooves_z
        self.outdir = outdir
        os.makedirs(self.outdir, exist_ok=True)

        self.lines = []       # finished polylines
        self.cur_pts = []     # current polyline
        self.undo_stack = []
        self.redo_stack = []

        (minx, miny, _), (maxx, maxy, _) = plate_mesh.bounds

        self.fig, self.ax = plt.subplots(figsize=(7, 7))
        self.ax.set_aspect("equal")
        self.ax.set_title("Draw polylines (Enter=finish, Ctrl+Z=undo, Ctrl+Y=redo, Generate=export)")
        self.ax.set_xlim(minx - 0.05*(maxx-minx), maxx + 0.05*(maxx-minx))
        self.ax.set_ylim(maxy + 0.05*(maxy-miny), miny - 0.05*(maxy-miny))

        # background preview aligned to mesh XY
        img = plt.imread(preview_image_path)
        self.ax.imshow(img, extent=[minx, maxx, maxy, miny], interpolation="bilinear")

        # Generate button
        ax_btn = self.fig.add_axes([0.78, 0.04, 0.18, 0.06])
        self.btn = Button(ax_btn, "Generate")
        self.btn.on_clicked(self.generate)

        # Events
        self.fig.canvas.mpl_connect("button_press_event", self.on_click)
        self.fig.canvas.mpl_connect("key_press_event", self.on_key)

        # Pan/zoom
        connect_default_panzoom(self.fig, self.ax)

        plt.show()

    def on_click(self, event):
        if event.inaxes != self.ax or event.button != 1:
            return
        pt = (float(event.xdata), float(event.ydata))
        self.cur_pts.append(pt)
        self.undo_stack.append(("point", pt))
        self.redo_stack.clear()
        self.draw_current()

    def on_key(self, event):
        if event.key in ("enter", "return"):
            if len(self.cur_pts) >= 2:
                polyline = list(self.cur_pts)
                self.lines.append(polyline)
                self.undo_stack.append(("line", polyline))
                self.redo_stack.clear()
                self.cur_pts = []
                self.redraw_lines()
        elif event.key == "ctrl+z":
            self.undo()
        elif event.key == "ctrl+y":
            self.redo()

    def undo(self):
        if not self.undo_stack:
            return
        action, value = self.undo_stack.pop()
        if action == "point":
            if self.cur_pts and self.cur_pts[-1] == value:
                self.cur_pts.pop()
        elif action == "line":
            if self.lines and self.lines[-1] == value:
                self.lines.pop()
        self.redo_stack.append((action, value))
        self.redraw_lines()
        self.draw_current()

    def redo(self):
        if not self.redo_stack:
            return
        action, value = self.redo_stack.pop()
        if action == "point":
            self.cur_pts.append(value)
        elif action == "line":
            self.lines.append(value)
        self.undo_stack.append((action, value))
        self.redraw_lines()
        self.draw_current()

    def draw_current(self):
        self.redraw_lines()
        if self.cur_pts:
            xs, ys = zip(*self.cur_pts)
            self.ax.plot(xs, ys, color="limegreen", lw=2)
            self.ax.scatter(xs, ys, color="royalblue", s=20)
            self.fig.canvas.draw_idle()

    def redraw_lines(self):
        [l.remove() for l in getattr(self, "_artists", [])] if hasattr(self, "_artists") else None
        self._artists = []
        for poly in self.lines:
            xs, ys = zip(*poly)
            a, = self.ax.plot(xs, ys, color="orange", lw=2, alpha=0.9)
            self._artists.append(a)
            b = self.ax.scatter(xs, ys, color="orange", s=20)
            self._artists.append(b)
        self.fig.canvas.draw_idle()

    def generate(self, _evt):
        if not self.lines:
            print("No cuts to apply.")
            return
        print("Generating OBJ pieces...")
        pieces = [self.mesh]
        for idx, polyline in enumerate(self.lines):
            print(f"  • Cutting with polyline {idx}: {polyline}")
            next_pieces = []
            for m in pieces:
                try:
                    comps = cut_mesh_with_polyline(m, polyline, kerf=self.kerf)
                    next_pieces.extend(comps if comps else [m])
                except Exception as e:
                    print(f"    ⚠️ Cut failed: {e}")
                    next_pieces.append(m)
            pieces = next_pieces

        # Apply grooves (with optional flip) and export each piece separately
        for i, p in enumerate(pieces):
            if self.grooves is not None:
                try:
                    p = add_grooves_to_piece(p, self.grooves, flip_grooves_z=self.flip_grooves_z)
                except Exception as e:
                    print(f"    ⚠️ Groove subtraction failed on piece {i}: {e}")

            # mirror Y for export (your workflow), then fix normals
            p.apply_scale([1, -1, -1])
            p.invert()
            path = os.path.join(self.outdir, f"piece_{i}.obj")
            p.export(path)
            print(f"  ✅ Exported {path}")

        print(f"Done. {len(pieces)} piece(s) written to {self.outdir}")


# ---------------- main ----------------

def main():
    ap = argparse.ArgumentParser(description="Interactive polyline cutter for country plate")
    ap.add_argument("--plate",   default="baseplates2_out/country_plate_no_country.obj", help="Base plate to cut (OBJ)")
    ap.add_argument("--grooves", default="baseplates2_out/country_plate_grooves.obj",    help="Groove cutter mesh (OBJ)")
    ap.add_argument("--preview", default="baseplates2_out/preview.png",                  help="Preview image aligned to mesh XY")
    ap.add_argument("--output",  default="countrypieces",                help="Output folder")
    ap.add_argument("--kerf",    type=float, default=0.3,                help="Cut thickness (mm)")
    ap.add_argument("--flip-grooves-z", action="store_true", default=False,
                    help="Mirror grooves across Z to apply on the opposite face (default on)")
    ap.add_argument("--no-flip-grooves-z", dest="flip_grooves_z", action="store_false")
    args = ap.parse_args()

    plate_mesh   = trimesh.load(args.plate,   force="mesh")
    grooves_mesh = trimesh.load(args.grooves, force="mesh") if os.path.exists(args.grooves) else None

    if not isinstance(plate_mesh, trimesh.Trimesh):
        raise RuntimeError("Base plate is not a triangle mesh")
    if grooves_mesh is not None and not isinstance(grooves_mesh, trimesh.Trimesh):
        grooves_mesh = None

    os.makedirs(args.output, exist_ok=True)
    SplitGUI(plate_mesh, grooves_mesh, args.preview, args.output, kerf=args.kerf,
             flip_grooves_z=args.flip_grooves_z)


if __name__ == "__main__":
    main()
