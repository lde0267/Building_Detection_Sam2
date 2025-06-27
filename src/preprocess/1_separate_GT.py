import os
import geopandas as gpd
from glob import glob
from shapely.geometry import Polygon, MultiPolygon

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))  # src의 부모 디렉토리 (즉, buildingDetection/)
DATA_DIR = os.path.join(BASE_DIR, "data")

digitalMap_folder = os.path.normpath(os.path.join(DATA_DIR, "each_digitalMap/each_gt"))
output_folder = os.path.normpath(os.path.join(DATA_DIR, "each_digitalMap/underseg_gt"))
margin = 0.6

os.makedirs(output_folder, exist_ok=True)

# SHP 파일 리스트 가져오기
shp_files = glob(os.path.join(digitalMap_folder, '*.shp'))

for shp_file in shp_files:
    gdf = gpd.read_file(shp_file)
    base_name = os.path.splitext(os.path.basename(shp_file))[0]
    counter = 1

    for _, row in gdf.iterrows():
        geom = row.geometry

        # MultiPolygon일 경우 각 폴리곤으로 분리
        if isinstance(geom, MultiPolygon):
             for poly in geom.geoms:
                new_row = row.copy()
                new_row.geometry = poly
                single_gdf = gpd.GeoDataFrame([new_row], columns=gdf.columns, crs=gdf.crs)
                output_path = os.path.join(output_folder, f"{base_name}_{counter}.shp")
                single_gdf.to_file(output_path)
                counter += 1
        elif isinstance(geom, Polygon):
            single_gdf = gpd.GeoDataFrame([row], columns=gdf.columns, crs=gdf.crs)
            output_path = os.path.join(output_folder, f"{base_name}_{counter}.shp")
            single_gdf.to_file(output_path)
            counter += 1
        else:
            print(f"지원하지 않는 geometry 유형: {type(geom)}")

print("모든 폴리곤 분리 완료!")