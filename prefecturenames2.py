import argparse
import os
import re
import trimesh
from jp2model import text_to_polygons, polygons_to_mesh


def safe_name(s: str) -> str:
    return re.sub(r"[^-\w]+", "_", s).strip("_")


def center_xy(mesh: trimesh.Trimesh) -> None:
    """Center XY only; leave Z unchanged (so we can control stacking)."""
    (minx, miny, _), (maxx, maxy, _) = mesh.bounds
    cx, cy = (minx + maxx) / 2.0, (miny + maxy) / 2.0
    mesh.apply_translation([-cx, -cy, 0.0])


def make_rect_mesh(width: float, height: float, thick: float, z_bottom: float) -> trimesh.Trimesh:
    """Rectangle centered at origin, bottom placed at z_bottom."""
    rect = trimesh.creation.box(extents=(width, height, thick))
    rect.apply_translation([0, 0, z_bottom + thick / 2.0])
    return rect


def build_kanji_geom(kanji: str, font_path: str, size: int, oversample: int):
    """Return polygons for kanji at full detail size (no XY scale yet)."""
    return text_to_polygons(
        kanji,
        font_path=font_path,
        fontsize=size,
        oversample=oversample,
    )


def rescale_mesh_xy(mesh: trimesh.Trimesh, factor: float):
    """Apply XY-only scaling, preserving Z thickness."""
    mesh.apply_scale([factor, factor, 1.0])


def make_outputs_for_one(
    kanji: str,
    out_dir: str,
    font_kanji: str,
    size: int,
    oversample: int,
    base_h: float,
    margin: float,
    piece_thickness: float,
    scale_xy: float,
    export_combined: bool,
    fit_clearance: float,
):
    os.makedirs(out_dir, exist_ok=True)

    # ---------- 1) Build kanji geometry (full detail, no scaling) ----------
    geom_kanji = build_kanji_geom(kanji, font_kanji, size, oversample)

    # Bounds at full size
    minx, miny, maxx, maxy = geom_kanji.bounds
    raw_w = (maxx - minx)
    raw_h = (maxy - miny)

    # ---------- 2) Target dimensions in FINAL (scaled) space ----------
    # These are the dimensions we want *after* kanji is scaled down,
    # so we bake scale_xy into W/H here and DO NOT scale the baseplates later.
    plate_w = raw_w * scale_xy + 2.0 * margin
    plate_h = raw_h * scale_xy + 2.0 * margin

    # Cutout gets extra clearance (per side) for easier fit
    cutout_w = plate_w + 2.0 * fit_clearance
    cutout_h = plate_h + 2.0 * fit_clearance

    # ---------- 3) Baseplate for printing (solid @ Z=0), already final size ----------
    baseplate_mesh = make_rect_mesh(width=plate_w, height=plate_h, thick=base_h, z_bottom=0.0)
    center_xy(baseplate_mesh)

    # ---------- 4) Kanji mesh: extrude full detail, then scale XY down ----------
    kanji_mesh = polygons_to_mesh(geom_kanji, height=base_h)
    center_xy(kanji_mesh)

    # Put kanji on top of baseplate
    minz_kanji = kanji_mesh.bounds[0][2]
    kanji_mesh.apply_translation([0, 0, base_h - minz_kanji])

    # Now shrink XY to target size (preserves tiny details)
    if scale_xy != 1.0:
        rescale_mesh_xy(kanji_mesh, scale_xy)

    # ---------- 5) Baseplate CUTOUT (negative), already final size ----------
    cutout_bottom = max(0.0, piece_thickness - base_h)
    baseplate_cutout_mesh = make_rect_mesh(
        width=cutout_w, height=cutout_h, thick=base_h, z_bottom=cutout_bottom
    )
    center_xy(baseplate_cutout_mesh)

    # ---------- 6) Export ----------
    base = safe_name(kanji) or "kanji"
    path_kanji = os.path.join(out_dir, f"{base}_kanji.obj")
    path_base = os.path.join(out_dir, f"{base}_kanji_baseplate.obj")
    path_cut = os.path.join(out_dir, f"{base}_kanji_baseplate_cutout.obj")

    kanji_mesh.export(path_kanji)
    baseplate_mesh.export(path_base)
    baseplate_cutout_mesh.export(path_cut)

    if export_combined:
        combo = trimesh.util.concatenate([baseplate_mesh.copy(), kanji_mesh.copy()])
        path_combo = os.path.join(out_dir, f"{base}_kanji_with_baseplate.obj")
        combo.export(path_combo)
        print(f"✅ {kanji}:")
        print(f"   - Kanji                → {path_kanji}")
        print(f"   - Baseplate (print)    → {path_base}")
        print(f"   - Baseplate (cutout)   → {path_cut}")
        print(f"   - Combined (stacked)   → {path_combo}")
    else:
        print(f"✅ {kanji}:")
        print(f"   - Kanji                → {path_kanji}")
        print(f"   - Baseplate (print)    → {path_base}")
        print(f"   - Baseplate (cutout)   → {path_cut}")


def load_prefecture_names(filepath: str):
    pairs = []
    with open(filepath, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            if "," in line:
                kanji, hira = line.split(",", 1)
                pairs.append((kanji.strip(), hira.strip()))
            else:
                pairs.append((line, ""))
    return pairs


def main():
    p = argparse.ArgumentParser(
        description="v2.5 – Detail-preserving kanji + baseplate + cutout "
                    "(XY scale after extrusion; margin/clearance in mm; baseplates not double-scaled)."
    )
    p.add_argument("--names-file", type=str, default="prefecture_names.txt",
                   help="UTF-8 file with lines 'KANJI,hiragana'")
    p.add_argument("--output", type=str, default="labels_v2",
                   help="Output folder for OBJ files.")
    p.add_argument("--font-kanji", type=str, default="NotoSansJP-Medium.ttf",
                   help="Font file for Kanji")
    p.add_argument("--size", type=int, default=800,
                   help="Font size for kanji rendering (big for detail)")
    p.add_argument("--oversample", type=int, default=4,
                   help="Supersampling factor")
    p.add_argument("--base-height", type=float, default=2.0,
                   help="Height of baseplate and kanji (mm)")
    p.add_argument("--margin", type=float, default=0.4,
                   help="Extra margin around kanji inside rectangle (mm)")
    p.add_argument("--piece-thickness", type=float, default=10.0,
                   help="Prefecture thickness (mm)")
    p.add_argument("--scale-xy", type=float, default=0.0075,
                   help="Uniform XY scaling factor applied AFTER extrusion")
    p.add_argument("--fit-clearance", type=float, default=0.1,
                   help="Extra clearance per side (mm) for baseplate cutout fit")
    p.add_argument("--no-combined", action="store_true",
                   help="Do NOT export the combined stacked OBJ")
    args = p.parse_args()

    names = load_prefecture_names(args.names_file)

    for kanji, _ in names:
        try:
            make_outputs_for_one(
                kanji=kanji,
                out_dir=args.output,
                font_kanji=args.font_kanji,
                size=args.size,
                oversample=args.oversample,
                base_h=args.base_height,
                margin=args.margin,
                piece_thickness=args.piece_thickness,
                scale_xy=args.scale_xy,
                export_combined=(not args.no_combined),
                fit_clearance=args.fit_clearance,
            )
        except Exception as e:
            print(f"❌ Failed for {kanji}: {e}")


if __name__ == "__main__":
    main()
