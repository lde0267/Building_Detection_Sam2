import os
import geopandas as gpd
import numpy as np
import cv2
from shapely.geometry import shape, Polygon
from shapely.ops import unary_union
from rasterio.features import rasterize
from rasterio.transform import from_origin

# 마스크를 폴리곤으로 변환
def mask_to_polygons(mask, transform):
    from rasterio.features import shapes
    from shapely.geometry import shape
    return [shape(geom) for geom, val in shapes(mask.astype(np.uint8), transform=transform) if val == 1]

# 가장 큰 폴리곤만 남기기
def filter_largest_polygon(polygons):
    if not polygons:
        return []
    return [max(polygons, key=lambda p: p.area)]

# 마스크 클린업 함수
def clean_building_mask(mask, kernel_size=5, iterations=1):
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
    opened = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=iterations)
    closed = cv2.morphologyEx(opened, cv2.MORPH_CLOSE, kernel, iterations=iterations)
    return closed

# shapefile 하나 처리
def clean_polygon_shapefile_without_tif(input_shp_path, output_shp_path, resolution=0.2):
    gdf = gpd.read_file(input_shp_path)
    if gdf.empty:
        print(f"[WARN] Empty shapefile: {input_shp_path}")
        return

    bounds = gdf.total_bounds  # xmin, ymin, xmax, ymax
    xmin, ymin, xmax, ymax = bounds

    # 가짜 transform 생성 (고정 해상도 기준)
    width = int(np.ceil((xmax - xmin) / resolution))
    height = int(np.ceil((ymax - ymin) / resolution))
    transform = from_origin(xmin, ymax, resolution, resolution)

    # Rasterize polygon
    mask = rasterize(
        [(geom, 1) for geom in gdf.geometry],
        out_shape=(height, width),
        transform=transform,
        fill=0,
        all_touched=True,
        dtype=np.uint8
    )

    # 모폴로지 연산
    cleaned_mask = clean_building_mask(mask)

    # 마스크 → 폴리곤
    polygons = mask_to_polygons(cleaned_mask, transform)
    filtered_polygons = filter_largest_polygon(polygons)

    if not filtered_polygons:
        print(f"[INFO] No valid polygon after cleaning: {input_shp_path}")
        return

    cleaned_gdf = gpd.GeoDataFrame(geometry=filtered_polygons, crs=gdf.crs)
    os.makedirs(os.path.dirname(output_shp_path), exist_ok=True)
    cleaned_gdf.to_file(output_shp_path)

    print(f"[✅] Saved cleaned shapefile: {output_shp_path}")


# === MAIN ===
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
DATA_DIR = os.path.join(BASE_DIR, "data")
OUTPUT_DIR = os.path.join(DATA_DIR, "output/result1")
CLEANED_OUTPUT_DIR = os.path.join(DATA_DIR, "output/result1_cleaned")

for shp_file in os.listdir(OUTPUT_DIR):
    if not shp_file.endswith(".shp"):
        continue

    input_shp_path = os.path.join(OUTPUT_DIR, shp_file)
    output_shp_path = os.path.join(CLEANED_OUTPUT_DIR, shp_file)

    clean_polygon_shapefile_without_tif(input_shp_path, output_shp_path)
