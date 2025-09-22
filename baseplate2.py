#!/usr/bin/env python3
import argparse
import os
import math
import trimesh
import shapely.geometry as geom
import shapely.ops as ops
import matplotlib.pyplot as plt


def flip_mesh(mesh: trimesh.Trimesh):
    """Flip mesh upside-down so grooves face upward in slicer."""
    mesh.apply_transform(trimesh.transformations.rotation_matrix(
        math.pi, [1, 0, 0], mesh.bounds.mean(axis=0)
    ))
    minz = mesh.bounds[0][2]
    mesh.apply_translation([0, 0, -minz])
    return mesh


def exclude_okinawa(country: trimesh.Trimesh):
    """Filter out Okinawa using a simple south cutoff."""
    minx, miny, minz = country.bounds[0]
    maxx, maxy, maxz = country.bounds[1]
    cutoff_y = miny + (maxy - miny) * 0.10  # heuristic: bottom ~10%
    keep_faces = []
    for i, face in enumerate(country.faces):
        verts = country.vertices[face]
        if verts[:, 1].mean() > cutoff_y:
            keep_faces.append(i)
    return country.submesh([keep_faces], append=True)


def country_outline(country_mesh, buffer_mm=20, simplify_mm=5):
    """Return simplified buffered outline polygon of the country mesh."""
    verts2d = country_mesh.vertices[:, :2]
    polys = [geom.Polygon(verts2d[face]) for face in country_mesh.faces]
    country_2d = ops.unary_union([p for p in polys if p.is_valid and not p.is_empty])
    outline = country_2d.buffer(buffer_mm).simplify(simplify_mm)
    return outline


def export_single_plate(outline_poly, outdir, thickness, groove_depth, country_mesh):
    os.makedirs(outdir, exist_ok=True)

    if outline_poly.geom_type == "MultiPolygon":
        outline_poly = max(outline_poly.geoms, key=lambda g: g.area)

    # Base plate extrusion
    plate = trimesh.creation.extrude_polygon(outline_poly, thickness)

    # Build clipping box to cut grooves
    minx, miny, maxx, maxy = outline_poly.bounds
    minz, maxz = country_mesh.bounds[:, 2]
    clip_box = trimesh.creation.box(
        extents=[maxx - minx, maxy - miny, (maxz - minz) + groove_depth]
    )
    clip_box.apply_translation([
        (minx + maxx) / 2.0,
        (miny + maxy) / 2.0,
        (minz + maxz) / 2.0
    ])

    try:
        clipped = country_mesh.intersection(clip_box)
    except Exception:
        clipped = None

    if not isinstance(clipped, trimesh.Trimesh) or clipped.is_empty:
        print("⚠️ Boolean failed, keeping flat plate.")
        plate_final = plate
        grooves = None
    else:
        grooves = clipped.copy()
        grooves.apply_translation([0, 0, -(maxz - groove_depth)])
        try:
            plate_final = plate.difference(grooves)
        except Exception:
            print("⚠️ Groove subtraction failed, flat plate only.")
            plate_final = plate

    # Flip everything for slicer
    plate_final = flip_mesh(plate_final)
    plate_no_country = flip_mesh(plate.copy())
    grooves = flip_mesh(grooves) if grooves is not None else None

    # Exports
    out_plate = os.path.join(outdir, "country_plate.obj")
    out_no_country = os.path.join(outdir, "country_plate_no_country.obj")
    out_grooves = os.path.join(outdir, "country_plate_grooves.obj")

    plate_final.export(out_plate)
    plate_no_country.export(out_no_country)
    if grooves:
        grooves.export(out_grooves)

    print(f"✅ Exported {out_plate}")
    print(f"✅ Exported {out_no_country}")
    if grooves:
        print(f"✅ Exported {out_grooves}")

    return outline_poly


def preview(outline_poly, country_mesh, outdir):
    """
    Save a PNG preview with:
      - exact cropping to the black outline bounds
      - white background (instead of transparent)
      - red fill for country, black outline
    """
    minx, miny, maxx, maxy = outline_poly.bounds

    fig, ax = plt.subplots(figsize=(8, 12))
    fig.patch.set_facecolor("white")   # white figure background
    ax.set_aspect("equal")

    # Draw outline (black)
    if outline_poly.geom_type == "Polygon":
        xs, ys = outline_poly.exterior.xy
        ax.plot(xs, ys, "k-", linewidth=2)
    elif outline_poly.geom_type == "MultiPolygon":
        for poly in outline_poly.geoms:
            xs, ys = poly.exterior.xy
            ax.plot(xs, ys, "k-", linewidth=2)

    # Draw country fill (red)
    verts2d = country_mesh.vertices[:, :2]
    faces = country_mesh.faces
    polys = [geom.Polygon(verts2d[face]) for face in faces]
    country_2d = ops.unary_union([p for p in polys if p.is_valid and not p.is_empty])
    if country_2d.geom_type == "Polygon":
        xs, ys = country_2d.exterior.xy
        ax.fill(xs, ys, color="red")
    elif country_2d.geom_type == "MultiPolygon":
        for poly in country_2d.geoms:
            xs, ys = poly.exterior.xy
            ax.fill(xs, ys, color="red")

    # Exact bounds
    ax.set_xlim(minx, maxx)
    ax.set_ylim(miny, maxy)
    ax.axis("off")

    fig.subplots_adjust(0, 0, 1, 1)   # remove figure margins

    out_path = os.path.join(outdir, "preview.png")
    plt.savefig(out_path, dpi=300, bbox_inches="tight", pad_inches=0, facecolor="white")
    plt.close()
    print(f"✅ Saved preview to {out_path}")



def main():
    parser = argparse.ArgumentParser(
        description="Generate outline baseplate with grooves and companion files."
    )
    parser.add_argument("--country", type=str, default="country_full.obj",
                        help="Path to country_full.obj")
    parser.add_argument("--output", type=str, default="baseplates2_out",
                        help="Output directory")
    parser.add_argument("--thickness", type=float, default=5.0,
                        help="Baseplate thickness in mm")
    parser.add_argument("--groove-depth", type=float, default=3.0,
                        help="Groove depth in mm")
    parser.add_argument("--buffer", type=float, default=20.0,
                        help="Outline buffer in mm")
    parser.add_argument("--simplify", type=float, default=5.0,
                        help="Simplify tolerance in mm")
    args = parser.parse_args()

    country = trimesh.load(args.country, force="mesh")
    if not isinstance(country, trimesh.Trimesh):
        raise RuntimeError("Country file is not a mesh")

    country = exclude_okinawa(country)

    minx, miny, minz = country.bounds[0]
    maxx, maxy, maxz = country.bounds[1]
    print(f"Country spans {maxx - minx:.1f} × {maxy - miny:.1f} mm")

    outline_poly = country_outline(country, buffer_mm=args.buffer, simplify_mm=args.simplify)
    outline_poly = export_single_plate(outline_poly, args.output, args.thickness, args.groove_depth, country)
    preview(outline_poly, country, args.output)


if __name__ == "__main__":
    main()
