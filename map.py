import os
import re
import zipfile
import requests
import argparse
import geopandas as gpd
from shapely.geometry import Polygon, MultiPolygon
from shapely import affinity
import trimesh


# =========================
# Helpers
# =========================
def safe_name(s: str) -> str:
    return re.sub(r"[^-\w]+", "_", s).strip("_")


def prepare_geometry(geom, name=""):
    """Return only the largest valid Polygon, or None if unusable."""
    if geom.is_empty:
        print(f"{name}: geometry empty, skipping.")
        return None

    # MultiPolygon → keep largest part only
    if isinstance(geom, MultiPolygon):
        print(f"{name}: MultiPolygon found, keeping largest part")
        geom = max(geom.geoms, key=lambda g: g.area)

    # Skip if still not a Polygon
    if not isinstance(geom, Polygon):
        print(f"{name}: not a Polygon, skipping.")
        return None

    # Try to fix invalid polygons
    if not geom.is_valid:
        try:
            geom = geom.buffer(0)
        except Exception as e:
            print(f"{name}: invalid geometry, skipping. Error: {e}")
            return None

    return geom


def extrude_any(geom, height_mm, name=""):
    """Extrude polygon into a 3D mesh."""
    if geom.is_empty or not isinstance(geom, Polygon):
        print(f"{name}: extrusion skipped (not a Polygon).")
        return None
    try:
        return trimesh.creation.extrude_polygon(geom, height_mm)
    except Exception as e:
        print(f"{name}: extrusion failed, retrying with buffer(0). Error: {e}")
        try:
            return trimesh.creation.extrude_polygon(geom.buffer(0), height_mm)
        except Exception as e2:
            print(f"{name}: extrusion completely failed. Error: {e2}")
            return None


# =========================
# Main
# =========================
def main(args):
    COUNTRY_ISO = args.country
    LEVEL = 1
    GADM_URL = f"https://geodata.ucdavis.edu/gadm/gadm4.1/shp/gadm41_{COUNTRY_ISO}_shp.zip"

    OUTPUT_DIR = args.output
    THICKNESS_MM = args.thickness
    CLEARANCE_MM = args.clearance
    MAP_WIDTH_MM = args.width
    SIMPLIFY_TOL = args.simplify_tolerance
    MAX_FACES = args.max_faces

    # Download dataset
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

    # Load shapefile
    shp_file = os.path.join(extract_dir, f"gadm41_{COUNTRY_ISO}_{LEVEL}.shp")
    gdf = gpd.read_file(shp_file, encoding="utf-8")
    gdf = gdf.to_crs(3857)  # meters

    # Scaling factors
    minx, miny, maxx, maxy = gdf.total_bounds
    width_m = maxx - minx
    scale_m_to_mm = MAP_WIDTH_MM / width_m
    tx_mm = -minx * scale_m_to_mm
    ty_mm = -miny * scale_m_to_mm

    # Output dir
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Find name field (prefer English)
    name_field = None
    for candidate in ["NAME_ENG", "NAME_1", "NAME", "NAME_JP"]:
        if candidate in gdf.columns:
            name_field = candidate
            break

    # Process subdivisions
    for idx, row in gdf.iterrows():
        name = row[name_field] if name_field else f"pref_{idx}"
        print(f"\n=== Processing {name} ===")

        # Step 1: get largest polygon
        geom_m = prepare_geometry(row.geometry, name)
        if geom_m is None:
            continue

        # Step 2: scale to mm
        geom_mm = affinity.scale(geom_m, xfact=scale_m_to_mm, yfact=scale_m_to_mm, origin=(0, 0))
        geom_mm = affinity.translate(geom_mm, xoff=tx_mm, yoff=ty_mm)

        # Step 3: clearance shrink
        if CLEARANCE_MM > 0:
            geom_mm = geom_mm.buffer(-CLEARANCE_MM)

        if geom_mm.is_empty:
            print(f"{name}: empty after clearance, skipping.")
            continue

        # Step 4: simplify
        if SIMPLIFY_TOL > 0:
            if isinstance(geom_mm, Polygon):
                before = len(geom_mm.exterior.coords)
            elif isinstance(geom_mm, MultiPolygon):
                before = "multi"
            else:
                before = "unknown"

            geom_mm = geom_mm.simplify(SIMPLIFY_TOL, preserve_topology=True)

            if isinstance(geom_mm, Polygon):
                after = len(geom_mm.exterior.coords)
            elif isinstance(geom_mm, MultiPolygon):
                after = "multi"
            else:
                after = "unknown"

            print(f"{name}: simplified tol={SIMPLIFY_TOL}, {before} → {after}")

        if geom_mm.is_empty:
            print(f"{name}: empty after simplify, skipping.")
            continue

        # Step 4b: ensure single polygon
        if isinstance(geom_mm, MultiPolygon):
            print(f"{name}: MultiPolygon after simplify/clearance, keeping largest part")
            geom_mm = max(geom_mm.geoms, key=lambda g: g.area)

        if not isinstance(geom_mm, Polygon):
            print(f"{name}: not a Polygon after cleanup, skipping.")
            continue

        # Step 5: extrusion
        print(f"{name}: extruding...")
        mesh = extrude_any(geom_mm, THICKNESS_MM, name=name)
        if mesh is None:
            print(f"{name}: extrusion failed, skipping.")
            continue
        print(f"{name}: extrusion done, faces={len(mesh.faces)}")

        # Step 6: decimation
        if MAX_FACES > 0 and len(mesh.faces) > MAX_FACES:
            if len(mesh.faces) > MAX_FACES * 20:
                print(f"{name}: too many faces ({len(mesh.faces)}), skipping decimation.")
            else:
                print(f"{name}: decimating from {len(mesh.faces)} → {MAX_FACES}...")
                mesh = mesh.simplify_quadratic_decimation(MAX_FACES)
                print(f"{name}: decimation done, faces={len(mesh.faces)}")

        # Step 7: place flat
        min_z = mesh.bounds[0][2]
        mesh.apply_translation([0, 0, -min_z])
        
        # After Step 7: place flat
        # New step: recenter XY
        # mesh.bounds is shape (2,3): [[minx, miny, minz], [maxx, maxy, maxz]]
        minx, miny, minz = mesh.bounds[0]
        maxx, maxy, maxz = mesh.bounds[1]

        # Compute center in XY plane
        cx = (minx + maxx) / 2
        cy = (miny + maxy) / 2

        # Move XY center to origin
        mesh.apply_translation([-cx, -cy, 0])

        # Step 8: export
        out_path = os.path.join(OUTPUT_DIR, f"{safe_name(name)}.obj")
        print(f"{name}: exporting {out_path}")
        mesh.export(out_path)
        print(f"{name}: export done")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate 3D puzzle pieces of country subdivisions.")
    parser.add_argument("--country", type=str, default="JPN", help="ISO country code (default: JPN)")
    parser.add_argument("--width", type=float, default=750.0, help="Total width of assembled map in mm (default: 750)")
    parser.add_argument("--thickness", type=float, default=10.0, help="Piece thickness in mm (default: 10)")
    parser.add_argument("--clearance", type=float, default=0.2, help="Clearance shrink in mm (default: 0.2)")
    parser.add_argument("--simplify-tolerance", type=float, default=0.1, help="Simplification tolerance in mm (default: 0.1)")
    parser.add_argument("--max-faces", type=int, default=5000, help="Maximum number of mesh faces after decimation (default: 5000)")
    parser.add_argument("--output", type=str, default="prefecture_objs", help="Output directory for OBJ files (default: prefecture_objs)")
    args = parser.parse_args()
    main(args)
