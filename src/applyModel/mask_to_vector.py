import cv2
import os
import numpy as np
import geopandas as gpd
from shapely.geometry import Polygon
import matplotlib.pyplot as plt
import rasterio
from prompt_generator import create_box
from apply_sam2 import generate_sam2_mask

# TIFF 영상에서 변환 정보, 이미지 가져오기
def load_tiff_with_metadata(tiff_path):
    """
    TIFF 이미지에서 RGB 이미지 배열, transform, crs 정보를 반환합니다.
    """
    with rasterio.open(tiff_path) as src:
        img_array = src.read([1, 2, 3])  # RGB 채널만
        img_array = np.transpose(img_array, (1, 2, 0)).astype(np.uint8)  # CHW → HWC
        transform = src.transform
        crs = src.crs
    return img_array, transform, crs
    
# 마스크 사이즈를 원본 TIF 크기에 맞게 조정하는 함수
def resize_mask_to_tif(mask, tif_path):

    if mask is None or mask.size == 0:
        print("Error : Empty mask array provided")
        return None
    # 원본 TIF 파일 크기 가져오기
    with rasterio.open(tif_path) as src:
        orig_width, orig_height = src.width, src.height

    # bool 타입이면 uint8로 변환
    if mask.dtype == bool:
        mask = mask.astype(np.uint8) * 255  # 0과 255로 변환 (시각화 편리)

    # 마스크 크기 조정 (원본 TIF 크기에 맞게)
    resized_mask = cv2.resize(mask, (orig_width, orig_height), interpolation=cv2.INTER_NEAREST)

    return resized_mask


# 마스크에서 폴리곤 추출
def mask_to_polygons(mask, transform):
    if mask is None or mask.size == 0:
        print("Error : Empty mask array provided")
        return None
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    polygons = []
    for contour in contours:
        if len(contour) >= 4:  # 최소 4개의 점이 필요
            geo_coords = [rasterio.transform.xy(transform, y, x) for x, y in contour[:, 0, :]]
            polygon = Polygon(geo_coords)
            polygons.append(polygon)
    return polygons

# 시각화
def visualize_mask_on_image(image, mask, box_coords=None, title="Mask Overlay"):

    if image.shape[:2] != mask.shape:
        mask = cv2.resize(mask.astype(np.uint8), (image.shape[1], image.shape[0]), interpolation=cv2.INTER_NEAREST)

    plt.figure(figsize=(10, 6))
    plt.imshow(image[..., ::-1])  # BGR → RGB

    # 마스크가 1인 부분만 lime 색으로 표시하고, 나머지는 투명하게 처리
    # mask==1 부분만 true로, 나머지는 false인 bool 마스크 생성
    mask_bool = mask.astype(bool)

    # numpy 배열에 색깔을 적용하려면 RGBA 형태로 만들어 투명도 조절
    # 아래는 이미지 크기와 동일한 RGBA 배열 생성 (모두 투명)
    overlay = np.zeros((mask.shape[0], mask.shape[1], 4), dtype=np.float32)

    # lime 색상 RGBA (lime: RGB=(0,1,0)), alpha=0.5 (반투명)
    overlay[mask_bool] = [0, 1, 0, 0.3]

    plt.imshow(overlay)

    if box_coords is not None:
        x0, y0, x1, y1 = box_coords.astype(int)
        plt.plot([x0, x1, x1, x0, x0],
                 [y0, y0, y1, y1, y0],
                 color='red', linestyle='--', linewidth=2, label='Box Prompt')
        plt.legend()

    plt.title(title)
    plt.axis("off")
    plt.show()



# 폴리곤을 Shapefile로 저장
def save_polygons_as_shapefile(polygons, crs, output_path):
    if not polygons:
        print("Warning: No polygons to save. Shapefile not created.")
        return
    gdf = gpd.GeoDataFrame({'geometry': polygons}, crs=crs)
    gdf.to_file(output_path)
    print(f"폴리곤 {len(polygons)}개를 검출하여 {output_path}에 저장했습니다.")

# 전체 실행 코드
def extract_polygons_from_sam(tiff_file, digit_file, output_file, predictor, margin=0.5):
    
    image, transform, crs = load_tiff_with_metadata(tiff_file)
    height, width, _ = image.shape

    # 0. create_box 함수로 박스 생성
    extended_box = create_box(digit_file, margin, tiff_file)

    # 1. SAM 마스크 생성
    mask = generate_sam2_mask(image, box_coords = extended_box, samPredictor = predictor)
    
    if mask is None or mask.sum() == 0:
        print(f"[INFO] No mask generated for {tiff_file}. Skipping.")
        return
    
    # 시각화 (마스크 + 박스)
    visualize_mask_on_image(
        image,
        mask,
        box_coords=extended_box,  # box도 함께 넘김
        title=os.path.basename(tiff_file) + " - SAM2 Mask"
    )

    # 마스크 사이즈 조정
    resized_mask = resize_mask_to_tif(mask, tiff_file)


    # 마스크를 TIFF 좌표계의 폴리곤으로 변환
    polygons = mask_to_polygons(resized_mask, transform)

    # 6. Shapefile 저장
    save_polygons_as_shapefile(polygons, crs, output_file)
