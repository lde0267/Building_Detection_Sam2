import rasterio
import geopandas as gpd
import pandas as pd
import numpy as np
from rasterio.windows import from_bounds
import os

# 폴더 경로 설정
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
DATA_DIR = os.path.join(BASE_DIR, "data")

digitalMap_folder = os.path.normpath(os.path.join(DATA_DIR, "each_digitalMap/each_gt"))
orthophoto_path = os.path.normpath(os.path.join(DATA_DIR, "origin_orthophoto/jungrang_Drone_Image_2022_5186.tif"))
output_folder = os.path.normpath(os.path.join(DATA_DIR, "each_orthophoto/each_ortho"))

# 디지털 폴리곤마다 반복
for digital_file in os.listdir(digitalMap_folder):
    if digital_file.endswith(".shp"):
        digital_file_path = os.path.join(digitalMap_folder, digital_file)

        # 디지털 폴리곤 불러오기
        polygons = gpd.read_file(digital_file_path)

        with rasterio.open(orthophoto_path) as src:
            crs = src.crs
            transform = src.transform

            # 좌표계 다르면 맞춰주기
            if polygons.crs != crs:
                polygons = polygons.to_crs(crs)
                print(f"📌 좌표계 변환 완료: {polygons.crs}")

            # 1. 합집합 영역 계산 (디지털 지도 기준)
            union_geometry = polygons.geometry.unary_union

            # 2. BBox 계산
            minx, miny, maxx, maxy = union_geometry.bounds

            # 3. 마진 추가
            # margin_ratio = 0.6
            # margin_x = min((maxx - minx) * margin_ratio, 10.0)
            # margin_y = min((maxy - miny) * margin_ratio, 10.0)

            # minx -= margin_x
            # maxx += margin_x
            # miny -= margin_y
            # maxy += margin_y

            # 3. 마진 추가 (절대값 8m)
            margin = 8.0

            minx -= margin
            maxx += margin
            miny -= margin
            maxy += margin

            # 4. 정사영상 crop
            window = from_bounds(minx, miny, maxx, maxy, transform)
            cropped_image = src.read(window=window)

            # 5. 변환 행렬 업데이트
            cropped_transform = src.window_transform(window)

        # 6. 저장
        output_filename = digital_file.replace("digitalPoly", "underSegOrtho").replace(".shp", ".tif")
        output_path = os.path.join(output_folder, output_filename)

        os.makedirs(output_folder, exist_ok=True)

        with rasterio.open(
            output_path,
            "w",
            driver="GTiff",
            height=cropped_image.shape[1],
            width=cropped_image.shape[2],
            count=cropped_image.shape[0],
            dtype=cropped_image.dtype,
            crs=src.crs,
            transform=cropped_transform,
        ) as dst:
            dst.write(cropped_image)

        print(f"✅ 디지털지도 '{digital_file}' 기준으로 생성된 정사영상이 '{output_path}'에 저장되었습니다.")
