import os
import re
import zipfile
import requests
import argparse
import geopandas as gpd
from shapely.geometry import Polygon, MultiPolygon
from shapely import affinity
import trimesh


def prepare_geometry(geom):
    """Return only the largest valid Polygon, or None if unusable."""
    if geom.is_empty:
        return None
    if isinstance(geom, MultiPolygon):
        geom = max(geom.geoms, key=lambda g: g.area)
    if not isinstance(geom, Polygon):
        return None
    if not geom.is_valid:
        try:
            geom = geom.buffer(0)
        except Exception:
            return None
    return geom


def extrude_polygon_safe(poly: Polygon, height_mm: float):
    try:
        return trimesh.creation.extrude_polygon(poly, height_mm)
    except Exception:
        try:
            return trimesh.creation.extrude_polygon(poly.buffer(0), height_mm)
        except Exception:
            return None


def main(args):
    COUNTRY_ISO = args.country
    LEVEL = 1  # prefecture level
    GADM_URL = f"https://geodata.ucdavis.edu/gadm/gadm4.1/shp/gadm41_{COUNTRY_ISO}_shp.zip"

    OUT_FILE = args.output
    MAP_WIDTH_MM = args.width
    THICKNESS_MM = args.thickness
    CLEARANCE_MM = args.clearance
    SIMPLIFY_TOL = args.simplify_tolerance

    # Download shapefile if not cached
    zip_path = f"gadm_{COUNTRY_ISO}.zip"
    if not os.path.exists(zip_path):
        print(f"Downloading {GADM_URL} ...")
        r = requests.get(GADM_URL)
        r.raise_for_status()
        with open(zip_path, "wb") as f:
            f.write(r.content)
        print("Download complete.")

    # Extract
    extract_dir = f"gadm_{COUNTRY_ISO}"
    if not os.path.exists(extract_dir):
        with zipfile.ZipFile(zip_path, "r") as z:
            z.extractall(extract_dir)
        print("Extraction complete.")

    # Load shapefile and project to meters
    shp_file = os.path.join(extract_dir, f"gadm41_{COUNTRY_ISO}_{LEVEL}.shp")
    gdf = gpd.read_file(shp_file, encoding="utf-8")
    gdf = gdf.to_crs(3857)

    # Scaling factors
    minx, miny, maxx, maxy = gdf.total_bounds
    width_m = maxx - minx
    scale_m_to_mm = MAP_WIDTH_MM / width_m
    tx_mm = -minx * scale_m_to_mm
    ty_mm = -miny * scale_m_to_mm

    meshes = []
    for _, row in gdf.iterrows():
        geom_m = prepare_geometry(row.geometry)
        if geom_m is None:
            continue

        # scale + translate to mm
        poly_mm = affinity.scale(geom_m, xfact=scale_m_to_mm, yfact=scale_m_to_mm, origin=(0, 0))
        poly_mm = affinity.translate(poly_mm, xoff=tx_mm, yoff=ty_mm)

        # Step 1: clearance shrink (match map.py)
        if CLEARANCE_MM > 0:
            poly_mm = poly_mm.buffer(-CLEARANCE_MM)
        if poly_mm.is_empty:
            continue

        # Step 2: simplify (match map.py)
        if SIMPLIFY_TOL > 0:
            poly_mm = poly_mm.simplify(SIMPLIFY_TOL, preserve_topology=True)
        if poly_mm.is_empty:
            continue

        # Step 3: re-grow to close seams
        if CLEARANCE_MM > 0:
            poly_mm = poly_mm.buffer(CLEARANCE_MM)
        if poly_mm.is_empty:
            continue

        # ensure polygon
        if isinstance(poly_mm, MultiPolygon):
            poly_mm = max(poly_mm.geoms, key=lambda g: g.area)
        if not isinstance(poly_mm, Polygon):
            continue

        # extrude
        mesh = extrude_polygon_safe(poly_mm, THICKNESS_MM)
        if mesh:
            meshes.append(mesh)

    if not meshes:
        raise RuntimeError("No prefecture meshes produced.")

    # Combine into one scene mesh
    country_mesh = trimesh.util.concatenate(meshes)

    # Center XY and place flat at Z=0
    minx, miny, minz = country_mesh.bounds[0]
    maxx, maxy, maxz = country_mesh.bounds[1]
    cx = (minx + maxx) / 2
    cy = (miny + maxy) / 2
    country_mesh.apply_translation([-cx, -cy, -minz])

    # Export
    country_mesh.export(OUT_FILE)
    print(f"✅ Exported full country mesh to {OUT_FILE}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate assembled country mesh (all prefectures in correct locations).")
    parser.add_argument("--country", type=str, default="JPN", help="ISO code (default: JPN)")
    parser.add_argument("--width", type=float, default=750.0, help="Total width of map in mm (default: 750)")
    parser.add_argument("--thickness", type=float, default=3.0, help="Extrusion height in mm")
    parser.add_argument("--clearance", type=float, default=0.2, help="Clearance shrink (default: 0.2)")
    parser.add_argument("--simplify-tolerance", type=float, default=0.1, help="Simplification tolerance (default: 0.1)")
    parser.add_argument("--output", type=str, default="country_full.obj", help="Output OBJ file")
    args = parser.parse_args()
    main(args)
