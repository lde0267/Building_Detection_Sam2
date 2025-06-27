import os
import sys
import glob
from mask_to_vector import extract_polygons_from_sam
from apply_sam2 import create_sam2_predictor_local

# 현재 파일(main.py)의 디렉토리 경로를 기준으로 상위 디렉토리를 sys.path에 추가
BASE_DIR = os.path.abspath(os.path.dirname(__file__))  
sys.path.append(BASE_DIR)


def main():
    # 현재 src 폴더 기준으로 data 폴더 경로 설정
    BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))  # src의 부모 디렉토리 (즉, Fastsam_ver2/)
    DATA_DIR = os.path.join(BASE_DIR, "data")

    ORTHO_DIR = os.path.join(DATA_DIR, "each_orthophoto/each_ortho")
    DIGIT_DIR = os.path.join(DATA_DIR, "each_digitalMap/each_gt")
    OUTPUT_DIR = os.path.join(DATA_DIR, "output/result1")
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # 모델 경로 설정
    checkpoint_path = os.path.join(BASE_DIR, "checkpoints/sam2.1_hiera_base_plus.pt")
    config_path = "configs/sam2.1/sam2.1_hiera_b+.yaml"

    predictor = create_sam2_predictor_local(config_path, checkpoint_path)
    print(type(predictor))

    # TIFF 파일 목록 가져오기 
    tif_files = glob.glob(os.path.join(ORTHO_DIR, "*.tif"))
    
    for tif_file in tif_files:
        # 파일명에서 번호 추출
        file_name = os.path.basename(tif_file)
        file_id = os.path.splitext(file_name)[0].replace("underSegOrtho", "")
        
        # 대응하는 폴리곤 파일과 출력 파일 설정
        digit_file = os.path.join(DIGIT_DIR, f"digitalPoly{file_id}.shp")
        output_file = os.path.join(OUTPUT_DIR, f"samPoly{file_id}.shp")
        print(tif_file)
        print(digit_file)

        # 파일 존재 여부 확인 후 실행
        if os.path.exists(digit_file):  
            print(f"Processing {tif_file} → {output_file}")
            extract_polygons_from_sam(tif_file, digit_file, output_file, predictor)
        else:
            print(f"Skipping {tif_file} (Missing {digit_file})")

if __name__ == "__main__":
    main()
