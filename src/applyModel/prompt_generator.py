import numpy as np
import geopandas as gpd
from rasterio.transform import rowcol
import rasterio

# polygon에서 margin 추가하여 EPSG 박스 → 이미지 픽셀 좌표 박스 변환 ---
def create_box(digitPath, margin, tiff_path):
    # polygon 읽기
    gdf = gpd.read_file(digitPath)
    # tif 메타정보 로드
    with rasterio.open(tiff_path) as src:
        transform = src.transform
        width = src.width
        height = src.height
        raster_crs = src.crs

    # 좌표계 맞추기
    if gdf.crs != raster_crs:
        gdf = gdf.to_crs(raster_crs)

    # 전체 polygon 경계 구하기
    xmin, ymin, xmax, ymax = gdf.total_bounds
    print(f"📦 EPSG Box: {np.array([xmin, ymin, xmax, ymax], dtype=np.float32)}")

    # margin 추가 (EPSG 좌표계 기준)
    xmin -= margin
    ymin -= margin
    xmax += margin
    ymax += margin

    # 좌표를 이미지 픽셀 좌표로 변환
    row_min, col_min = rowcol(transform, xmin, ymax)  # 좌상단 (xmin, ymax)
    row_max, col_max = rowcol(transform, xmax, ymin)  # 우하단 (xmax, ymin)

    # 좌표 정렬 및 클리핑
    x_min = max(0, min(col_min, col_max))
    x_max = min(width - 1, max(col_min, col_max))
    y_min = max(0, min(row_min, row_max))
    y_max = min(height - 1, max(row_min, row_max))
    
    print(f"📸 Pixel Box: {np.array([x_min, y_min, x_max, y_max], dtype=np.float32)}")

    return np.array([x_min, y_min, x_max, y_max], dtype=np.float32)