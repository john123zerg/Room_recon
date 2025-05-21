import os
import torch
import numpy as np
from PIL import Image
import sys
import os
utils_path = os.path.join(os.path.dirname(__file__), 'Grounded-SAM-2', 'utils')

sys.path.append(utils_path)
utils_path = os.path.join(os.path.dirname(__file__), 'Grounded-SAM-2', 'sam-hq')
sys.path.append(utils_path)
from sam2.build_sam import build_sam2_hq_video_predictor, build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor
from video_utils import create_video_from_images
from common_utils import CommonUtils
from mask_dictionary_model import MaskDictionaryModel, ObjectInfo
import json
import copy

# 추가: natsort 라이브러리 임포트
try:
    from natsort import natsorted
except ImportError:
    print("natsort 라이브러리가 설치되어 있지 않습니다. pip install natsort로 설치해주세요.")
    # 대체 구현 (natsort가 없을 경우)
    import re
    def natural_sort_key(s):
        return [int(text) if text.isdigit() else text.lower() for text in re.split(r'(\d+)', s)]
    
    def natsorted(l):
        return sorted(l, key=natural_sort_key)

import torch


# init sam image predictor and video predictor model
sam2_checkpoint = "sam-hq/sam-hq2/checkpoints/sam2.1_hq_hiera_large.pt"
model_cfg = "configs/sam2.1/sam2.1_hq_hiera_l.yaml"
device = "cuda" if torch.cuda.is_available() else "cpu"
print("device", device)

video_predictor = build_sam2_hq_video_predictor(model_cfg, sam2_checkpoint)
sam2_image_model = build_sam2(model_cfg, sam2_checkpoint, device=device)
image_predictor = SAM2ImagePredictor(sam2_image_model)

import os
import sys
import re

gsam2_path = os.path.join(os.path.dirname(__file__), "Grounded-SAM-2")

sys.path.append(gsam2_path)  # 또는 sys.path.insert(0, gsam2_path)

# Replace Hugging Face API-based Grounding DINO initialization with local model loading
from grounding_dino.groundingdino.util.inference import load_model, predict
from grounding_dino.groundingdino.util import box_ops

# Load the Grounding DINO model locally
grounding_dino_config = "Grounded-SAM-2/grounding_dino/groundingdino/config/GroundingDINO_SwinT_OGC.py"
grounding_dino_checkpoint = "Grounded-SAM-2/gdino_checkpoints/groundingdino_swint_ogc.pth"
grounding_model = load_model(
    model_config_path=grounding_dino_config,
    model_checkpoint_path=grounding_dino_checkpoint,
    device=device
)
# Ensure the model is moved to the correct device
grounding_model = grounding_model.to(device)  # Explicitly move the model to the correct device

# setup the input image and text prompt for SAM 2 and Grounding DINO
# VERY important: text queries need to be lowercased + end with a dot
text = "a chair, a table, floor, wall."

# `video_dir` a directory of JPEG frames with filenames like `<frame_index>.jpg`  
# video_dir = "notebooks/videos/car"
video_dir = os.path.abspath("datasets/images")

# 'output_dir' is the directory to save the annotated frames
output_dir = "./outputs"
# 'output_video_path' is the path to save the final video
output_video_path = "./outputs/output.mp4"
# create the output directory
CommonUtils.creat_dirs(output_dir)
mask_data_dir = os.path.join(output_dir, "mask_data")
json_data_dir = os.path.join(output_dir, "json_data")
result_dir = os.path.join(output_dir, "result")
CommonUtils.creat_dirs(mask_data_dir)
CommonUtils.creat_dirs(json_data_dir)
CommonUtils.creat_dirs(result_dir)  # Make sure result_dir is created
# scan all the JPEG frame names in this directory
frame_names = [
    p for p in os.listdir(video_dir)
    if os.path.splitext(p)[-1] in [".jpg", ".jpeg", ".JPG", ".JPEG", ".png", ".PNG"]
]
print(f'frame_names: {frame_names}')


frame_names = natsorted(frame_names)

# init video predictor state
inference_state = video_predictor.init_state(video_path=video_dir)
step = 25 # the step to sample frames for Grounding DINO predictor

sam2_masks = MaskDictionaryModel()

PROMPT_TYPE_FOR_VIDEO = "mask" # box, mask or point
objects_count = 0

"""
Step 2: Prompt Grounding DINO and SAM image predictor to get the box and mask for all frames
"""
from torchvision import transforms

print("Total frames:", len(frame_names))

from torchvision.ops import box_iou

def merge_overlapping_boxes(input_boxes,confidences, class_names, iou_thresh=0.4):
    boxes = torch.tensor(input_boxes).float()
    ious = box_iou(boxes, boxes)
    
    used = set()
    groups = []
    for i in range(len(boxes)):
        if i in used:
            continue
        group = [i]
        used.add(i)
        for j in range(i + 1, len(boxes)):
            if j not in used and ious[i, j] > iou_thresh:
                group.append(j)
                used.add(j)
        groups.append(group)

    merged_boxes = []
    merged_labels = []
    merged_scores = []

    for group in groups:
        group_boxes = boxes[group]
        x1 = torch.min(group_boxes[:, 0])
        y1 = torch.min(group_boxes[:, 1])
        x2 = torch.max(group_boxes[:, 2])
        y2 = torch.max(group_boxes[:, 3])
        merged_boxes.append([x1.item(), y1.item(), x2.item(), y2.item()])

        group_scores = [confidences[i] for i in group]
        group_labels = [class_names[i] for i in group]
        merged_scores.append(max(group_scores))
        merged_labels.append(group_labels[0])  # 가장 앞의 label 사용

    return merged_boxes, merged_scores, merged_labels


for start_frame_idx in range(0, len(frame_names), step):
    # Prompt Grounding DINO to get the box coordinates on a specific frame
    print("start_frame_idx", start_frame_idx)
    img_path = os.path.join(video_dir, frame_names[start_frame_idx])
    pil_image = Image.open(img_path).convert("RGB")
    image= Image.open(img_path).convert("RGB")
    image_base_name = frame_names[start_frame_idx].split(".")[0]
    mask_dict = MaskDictionaryModel(promote_type = PROMPT_TYPE_FOR_VIDEO, mask_name = f"mask_{image_base_name}.npy")
    from torchvision.io import read_image

    image_tensor = read_image(img_path).to(device).float() / 255.0

    print(f"Image tensor shape: {image_tensor.shape}")
    # Pass the PyTorch tensor directly to the predict function
    boxes, conf, labels = predict(
        model=grounding_model,
        image=image_tensor,  # Pass unbatched tensor (shape: [3, H, W])
        caption=text,
        box_threshold=0.3,
        text_threshold=0.1,
    )

   
    boxes_xyxy=[]            
    # Convert tensor image back to PIL image for drawing
    image_np = (image_tensor.cpu().numpy() * 255).astype(np.uint8)  # shape: [3, H, W]
    image_pil = Image.fromarray(np.transpose(image_np, (1, 2, 0)))  # to [H, W, 3]
    
    for box, label, score in zip(boxes, labels, conf):
        cx, cy, w, h = box.tolist()
        x1 = (cx - w / 2) * image_pil.size[0]
        x2 = (cx + w / 2) * image_pil.size[0]
        y1 = (cy - h / 2) * image_pil.size[1]
        y2 = (cy + h / 2) * image_pil.size[1]
        x1, x2 = sorted([x1, x2])
        y1, y2 = sorted([y1, y2])
        box = [x1, y1, x2, y2]
        boxes_xyxy.append(box)
    print(f'boxes_xyxy: {boxes_xyxy}')
    if len(boxes_xyxy) == 0:
        boxes_xyxy = np.array([])
    else:
        boxes_xyxy,conf,labels= merge_overlapping_boxes(boxes_xyxy, conf, labels, iou_thresh=0.5)
    print(f'boxes: {boxes_xyxy}')


    input_boxes=[]
    confidences=[]
    class_names=[]
    for i, (box, conf, label) in enumerate(zip(boxes_xyxy, conf, labels)):
        print(f'  {i+1}: {label} (confidence: {conf:.4f}, box: {box})') 
  
        input_boxes.append(box)
        confidences.append(conf)
        class_names.append(label)
    
    
    input_boxes=np.array(input_boxes)    
    OBJECTS = labels
    print(f'input_boxes: {input_boxes}')
    print(f'OBJECTS: {OBJECTS}')

    if input_boxes.shape[0] != 0:
        # prompt SAM image predictor to get the mask for the object
        image_predictor.set_image(np.array(image.convert("RGB")))

        masks, scores, logits = image_predictor.predict(
            point_coords=None,
            point_labels=None,
            box=input_boxes,
            multimask_output=False,
        )

        if masks.ndim == 2:
            masks = masks[None]
            scores = scores[None]
            logits = logits[None]
        elif masks.ndim == 4:
            masks = masks.squeeze(1)

        if mask_dict.promote_type == "mask":
            mask_dict.add_new_frame_annotation(
                mask_list=torch.tensor(masks).to(device),
                box_list=torch.tensor(input_boxes),
                label_list=OBJECTS
            )
        else:
            raise NotImplementedError("SAM 2 video predictor only support mask prompts")

        objects_count = mask_dict.update_masks(
            tracking_annotation_dict=sam2_masks,
            iou_threshold=0.8,
            objects_count=objects_count
        )
        print("objects_count", objects_count)
        print(f'score: {scores}')

    else:
        print(f"No object detected in the frame, skip merge the frame {frame_names[start_frame_idx]}")
        mask_dict = sam2_masks

    # Step 4: Propagate masks
    if len(mask_dict.labels) == 0:
        mask_dict.save_empty_mask_and_json(
            mask_data_dir,
            json_data_dir,
            image_name_list=frame_names[start_frame_idx:start_frame_idx+step]
        )
        print(f"No object detected in the frame, skip the frame {start_frame_idx}")
        continue
    else:
        video_predictor.reset_state(inference_state)

        for object_id, object_info in mask_dict.labels.items():
            frame_idx, out_obj_ids, out_mask_logits = video_predictor.add_new_mask(
                inference_state,
                start_frame_idx,
                object_id,
                object_info.mask,
            )

        video_segments = {}
        for out_frame_idx, out_obj_ids, out_mask_logits in video_predictor.propagate_in_video(
            inference_state,
            max_frame_num_to_track=step,
            start_frame_idx=start_frame_idx
        ):
            frame_masks = MaskDictionaryModel()
            for i, out_obj_id in enumerate(out_obj_ids):

                obj_logits = out_mask_logits[i]
                

                if out_frame_idx == start_frame_idx:

                    out_mask = (obj_logits > 0.0)
                else:
         
                    normalized_logits = torch.sigmoid(obj_logits)
                    out_mask = (normalized_logits > 0.3)
                
                print(f"Frame {out_frame_idx}, Object {out_obj_id}: Mask sum = {out_mask.sum()}")
                
                object_info = ObjectInfo(
                    instance_id=out_obj_id,
                    mask=out_mask.squeeze(), 
                    class_name=mask_dict.get_target_class_name(out_obj_id)
                )
                object_info.update_box()
                frame_masks.labels[out_obj_id] = object_info
                image_base_name = frame_names[out_frame_idx].split(".")[0]
                frame_masks.mask_name = f"mask_{image_base_name}.npy"
                frame_masks.mask_height = out_mask.shape[-2]
                frame_masks.mask_width = out_mask.shape[-1]

            video_segments[out_frame_idx] = frame_masks
            sam2_masks = copy.deepcopy(frame_masks)

        print("video_segments:", len(video_segments))

    # Step 5: Save masks and metadata
    for frame_idx, frame_masks_info in video_segments.items():
        print(f'frame_idx', frame_idx)
        mask = frame_masks_info.labels
        mask_img = torch.zeros(frame_masks_info.mask_height, frame_masks_info.mask_width)

        for obj_id, obj_info in mask.items():
            mask_img[obj_info.mask == True] = obj_id

        mask_img = mask_img.numpy().astype(np.uint16)
        
        np.save(os.path.join(mask_data_dir, frame_masks_info.mask_name), mask_img)

        json_data = frame_masks_info.to_dict()
        print(f'json_data', json_data)
        json_data_path = os.path.join(json_data_dir, frame_masks_info.mask_name.replace(".npy", ".json"))
        with open(json_data_path, "w") as f:
            json.dump(json_data, f)



CommonUtils.draw_masks_and_box_with_supervision(
    video_dir,
    mask_data_dir,
    json_data_dir,
    result_dir,
    #stick_to_raw_resolution=True
)
result_files = natsorted([f for f in os.listdir(result_dir) if f.endswith((".png", ".jpg"))])
for idx, fname in enumerate(result_files):
    old_path = os.path.join(result_dir, fname)
    new_path = os.path.join(result_dir, f"{idx:05d}.png")
    os.rename(old_path, new_path)
    print(f'Renamed {old_path} to {new_path}')f
create_video_from_images(result_dir, output_video_path, frame_rate=30)

