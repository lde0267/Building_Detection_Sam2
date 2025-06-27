# Building Detection Using SAM2 with Orthophotos and Digital Maps

## 프로젝트 개요
이 프로젝트는 정사영상(Orthophoto)과 수치지도를 기반으로, SAM2(Segment Anything Model 2)를 활용해 건물 객체를 자동으로 추출하는 워크플로우를 구현합니다.  
수치지도의 폴리곤 정보를 바탕으로 이미지 내 관심 영역(Box Prompt)를 생성하고, 이를 SAM2에 입력하여 건물 마스크를 생성 및 후처리하여 최종 건물 객체를 얻습니다.

## 주요 기능
- TIFF 정사영상 로드 및 메타데이터(좌표변환, CRS) 추출  
- 수치지도의 폴리곤(Shapefile) 로드 및 좌표계 변환 후 이미지 좌표계로 변환  
- 폴리곤 경계에 마진을 추가하여 Box Prompt 생성  
- SAM2 모델을 통한 건물 마스크 예측  
- 마스크 후처리: 모폴로지 연산을 이용한 노이즈 제거 및 가장 큰 건물 폴리곤만 추출  
- 최종 폴리곤 Shapefile 저장  
- 시각화 기능: 이미지 위에 건물 마스크와 Box Prompt 오버레이 표시  

## 폴더 구조 예시
├─ checkpoints/
│ └─ sam2.1_hiera_base_plus.pt # SAM2 모델 체크포인트
├─ data/
│ ├─ each_orthophoto/each_ortho/ # TIFF 정사영상
│ ├─ each_digitalMap/each_gt/ # 수치지도 폴리곤 Shapefile
│ └─ output/
│   ├─ result1/ # SAM2 결과 저장 폴더
│   └─ result1_cleaned/ # 후처리 완료 폴더
└─ src/
├─ applyModel/ # 모델 적용 및 처리 스크립트
├─ preprocess/ # 전처리 스크립트
└─ postprocess/ # 후처리 스크립트

## 주요 패키지
rasterio (TIFF 영상 처리 및 좌표 변환)
geopandas (벡터 지리 정보 처리)
shapely (지오메트리 처리)
numpy
matplotlib (시각화)
opencv-python (이미지 후처리)
torch (PyTorch, SAM2 모델 실행)

SAM2는 OpenAI 또는 Meta에서 공개한 체크포인트 및 코드 베이스를 별도로 설치해야 하며, 아래 경로에 체크포인트가 있어야 합니다:
checkpoints/sam2.1_hiera_base_plus.pt

# 사용법 요약
1. preprocess 실행을 통해 영상 크롭 및 폴리곤 개별화
2. `src/applyModel/main.py` 를 실행하여 전체 모델 적용 pipeline 동작
3. postprocess 실행을 통해 모폴로지 연산 및 작은 객체 삭제

## 주요 파라미터
- TIFF 정사영상 경로: `data/each_orthophoto/each_ortho/`
- 수치지도 Shapefile 경로: `data/each_digitalMap/each_gt/`
- 출력 폴더: `data/output/result1/` 및 후처리 결과 `data/output/result1_cleaned/`

## 프로세스 순서
1. TIFF와 대응하는 수치지도 Shapefile에서 폴리곤을 읽고 EPSG 좌표계에서 이미지 픽셀 좌표계로 변환  
2. 변환된 좌표에 margin을 더해 Box Prompt 생성  
3. Box Prompt를 SAM2에 입력하여 건물 마스크 생성  
4. 마스크에 모폴로지 연산 등 후처리 적용  
5. 후처리된 마스크를 다시 폴리곤으로 변환하여 Shapefile로 저장  

## 주요 함수 및 모듈
- `load_image(tiff_path)`: TIFF 영상 읽기 및 numpy 배열 반환  
- `create_box2(digit_path, margin, tiff_path)`: 수치지도 폴리곤에서 margin 포함 Box Prompt 생성  
- `generate_sam2_mask(image, box_coords, samPredictor)`: SAM2를 활용해 Box Prompt 기반 마스크 생성  
- `clean_building_mask(mask, kernel_size, iterations)`: 모폴로지 연산으로 노이즈 제거 및 외곽 다듬기  
- `mask_to_polygons(mask, transform)`: 마스크를 벡터 폴리곤으로 변환  
- `save_polygons_as_shapefile(polygons, crs, output_path)`: 폴리곤을 Shapefile로 저장  
- `visualize_mask_on_image(image, mask, box_coords, title)`: 결과 시각화  

