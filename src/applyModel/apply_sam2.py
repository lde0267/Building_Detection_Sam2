import numpy as np
from rasterio.transform import rowcol
import torch
from sam2.sam2_image_predictor import SAM2ImagePredictor
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor

def create_sam2_predictor_local(config_path, checkpoint_path, device="cuda" if torch.cuda.is_available() else "cpu"):
    # 1. build_sam2 함수를 이용해 모델 생성 (config + checkpoint)
    model = build_sam2(config_path, checkpoint_path)
    model.to(device)
    
    # 2. SAM2ImagePredictor에 모델 할당
    predictor = SAM2ImagePredictor(model)
    
    return predictor

# ✅ 3. 마스크 병합
def combine_masks(mask_list):
    if len(mask_list) == 0:
        return None
    combined_mask = np.zeros_like(mask_list[0])
    for mask in mask_list:
        combined_mask = np.logical_or(combined_mask, mask).astype(np.uint8)
    return combined_mask


# ✅ 4. SAM2로 마스크 생성 (박스 프롬프트 사용)
from PIL import Image
import numpy as np
import torch
import rasterio

def generate_sam2_mask(image, box_coords, samPredictor):
    device = "cuda" if torch.cuda.is_available() else "cpu"

    with torch.inference_mode(), torch.autocast(device, dtype=torch.bfloat16):
        samPredictor.set_image(image)
        masks, scores, logits = samPredictor.predict(
            box=box_coords.tolist(),
            multimask_output=False
        )
    if masks is not None and len(masks) > 0:
        mask_array = masks[0].cpu().numpy() if hasattr(masks[0], "cpu") else masks[0]
        return mask_array.astype(bool)
    else:
        print("No mask detected")
        return None
