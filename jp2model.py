import argparse
import os
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from shapely.geometry import Polygon, MultiPolygon
from shapely import affinity
from shapely.ops import unary_union
import trimesh
from skimage import measure  # kept for optional debugging; not required in final logic
import cv2


def text_to_polygons(text, font_path, fontsize=200, oversample=4):
    font_path = os.path.abspath(font_path)
    font = ImageFont.truetype(font_path, fontsize * oversample)

    # Oversized canvas (avoid clipping descenders)
    W, H = fontsize * oversample * len(text) * 2, fontsize * oversample * 3
    img = Image.new("L", (W, H), 0)
    draw = ImageDraw.Draw(img)

    # Draw text with padding from top
    pad_x, pad_y = fontsize * oversample, fontsize * oversample
    draw.text((pad_x, pad_y), text, font=font, fill=255)

    mask = np.array(img)

    # Auto-crop to content (find nonzero pixels)
    ys, xs = np.nonzero(mask)
    minx, maxx = xs.min(), xs.max()
    miny, maxy = ys.min(), ys.max()
    mask = mask[miny:maxy+1, minx:maxx+1]

    # Find contours with hierarchy
    contours, hierarchy = cv2.findContours(mask, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
    if hierarchy is None:
        raise RuntimeError("No contours found. Check font/text support.")

    H = mask.shape[0]
    polygons = []

    for i, cnt in enumerate(contours):
        parent = hierarchy[0][i][3]
        if parent != -1:
            continue

        ext = contours[i][:, 0, :].astype(float)
        exterior = [(x, H - y) for x, y in ext]

        holes = []
        child = hierarchy[0][i][2]
        while child != -1:
            ring = contours[child][:, 0, :].astype(float)
            holes.append([(x, H - y) for x, y in ring])
            child = hierarchy[0][child][0]

        poly = Polygon(exterior, holes).buffer(0)
        if poly.is_valid and not poly.is_empty:
            polygons.append(poly)

    geom = MultiPolygon(polygons) if len(polygons) > 1 else polygons[0]
    # Scale back down
    geom = affinity.scale(geom, xfact=1/oversample, yfact=1/oversample, origin=(0, 0))
    return geom



def polygons_to_mesh(geom, height=3.0):
    """Extrude shapely (Multi)Polygon(s) into a single trimesh mesh."""
    if isinstance(geom, MultiPolygon):
        polys = list(geom.geoms)
    else:
        polys = [geom]

    meshes = []
    for poly in polys:
        if poly.area < 1:  # skip noise
            continue
        meshes.append(trimesh.creation.extrude_polygon(poly, height))

    if not meshes:
        raise RuntimeError("No valid geometry to extrude.")
    return trimesh.util.concatenate(meshes)


def add_baseplate(mesh, geom, thickness=2.0, margin=10.0):
    """Add a rectangular baseplate under the text."""
    minx, miny, maxx, maxy = geom.bounds
    width, height = maxx - minx, maxy - miny
    base = trimesh.creation.box(extents=(width + margin, height + margin, thickness))
    base.apply_translation([minx + width / 2, miny + height / 2, -thickness])
    return trimesh.util.concatenate([base, mesh])


def main():
    parser = argparse.ArgumentParser(description="Convert Japanese text to a 3D model (OBJ/STL).")
    parser.add_argument("text", help="Japanese text to convert")
    parser.add_argument("-o", "--output", default="label.obj", help="Output file (extension decides format: .obj/.stl/...)")
    parser.add_argument("-f", "--font", default="NotoSansJP-Regular.ttf", help="Path to a Japanese-capable .ttf/.ttc font")
    parser.add_argument("--size", type=int, default=200, help="Font size (rendering scale)")
    parser.add_argument("--height", type=float, default=3.0, help="Extrusion height (mm)")
    parser.add_argument("--base", action="store_true", help="Add rectangular baseplate under text")
    parser.add_argument("--base_thickness", type=float, default=2.0, help="Baseplate thickness (mm)")
    parser.add_argument("--base_margin", type=float, default=10.0, help="Extra margin around text (mm)")
    parser.add_argument("--oversample", type=int, default=4, help="Supersampling factor for smoother curves")
    args = parser.parse_args()

    geom = text_to_polygons(args.text, font_path=args.font, fontsize=args.size, oversample=args.oversample)
    mesh = polygons_to_mesh(geom, height=args.height)

    if args.base:
        mesh = add_baseplate(mesh, geom, thickness=args.base_thickness, margin=args.base_margin)

    mesh.export(args.output)
    print(f"✅ Saved {args.output}")


if __name__ == "__main__":
    main()
