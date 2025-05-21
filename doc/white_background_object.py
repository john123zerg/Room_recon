import os
import cv2
import numpy as np
import natsort

original_image_path = "datasets/images/"
mask_data_dir = "./outputs/mask_data/"
output_dir = "./outputs/merged_images/"


if not os.path.exists(output_dir):
    os.makedirs(output_dir)

frame_num = len([f for f in os.listdir(original_image_path) if f.endswith(('.png', '.jpg', '.jpeg', '.PNG', '.JPG', '.JPEG'))])
full_image_list = os.listdir(original_image_path)
import re

def extract_number(x):
    numbers = re.findall(r'\d+', x)
    return int(numbers[-1]) if numbers else float('inf')  # 숫자 없으면 맨 뒤로
# 파일 목록을 정렬하고 베이스 이름 추출하는 부분을 수정
full_image_list = [f for f in os.listdir(original_image_path) if os.path.splitext(f)[-1].lower() in ('.png', '.jpg', '.jpeg')]
full_image_list = natsort.natsorted(full_image_list, key=extract_number)

# 파일명과 베이스명의 매핑 만들기 (더 안전한 방법)
full_name_to_file_map = {os.path.splitext(f)[0]: f for f in full_image_list}
full_name_list = list(full_name_to_file_map.keys())

print(f'full_image_list: {full_image_list}')
print(f'full_name_list: {full_name_list}')
print(f"Total frames found: {len(full_image_list)}")

print(f'full_image_list: {full_image_list}')
print(f"Total frames found: {frame_num}")



json_data_dir = "./outputs/json_data/"


target_class = "a chair, a table, floor, wall."
import os
import json
from collections import defaultdict
# 결과: {"frame_id1": [1, 3], "frame_id2": [5], ...}
class_instance_ids_per_frame = defaultdict(list)

for filename in os.listdir(json_data_dir):
    if filename.endswith(".json"):
        print(f'Processing {filename}...')
        frame_id = filename
        print(f'frame_id: {frame_id}')
        if frame_id is None:
            print(f"⚠️  No numeric frame_id found in {filename}, skipping.")
            continue

        json_path = os.path.join(json_data_dir, filename)
        print(f'json_path: {json_path}')
        with open(json_path, "r") as f:
            data = json.load(f)
        print(f'data: {data}')
        labels = data.get("labels", {})
        for inst_info in labels.values():
            if inst_info["class_name"] in target_class:
                class_instance_ids_per_frame[frame_id].append(inst_info["instance_id"])





object_list = []

print(f'class_instance_ids_per_frame: {class_instance_ids_per_frame}')
all_instance_ids = sorted(set(
    instance_id
    for ids in class_instance_ids_per_frame.values()
    for instance_id in ids
))

print(f'all_instance_ids: {all_instance_ids}')

for selected_id in all_instance_ids:
    print(f'selected_id: {selected_id}')
    images_for_video = []  
    for i in full_name_list:
        original_file = [f for f in full_image_list if f.startswith(f"{i}.")][0]
        image_name = os.path.join(original_image_path, original_file)
        
        mask_name = os.path.join(mask_data_dir, f"mask_{i}.npy")
        print(f'image_name: {image_name}')
        print(f'mask_name: {mask_name}')
        
        mask_img = np.load(mask_name)
        mask_img = mask_img.astype(np.uint16)
        original_image = cv2.imread(image_name)
        original_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)

        merged_image = np.ones_like(original_image) * 255 
        merged_image[mask_img == selected_id] = original_image[mask_img == selected_id]

        if not os.path.exists(os.path.join(f"./outputs/merged_images/{selected_id}")):
            os.makedirs(os.path.join(f"./outputs/merged_images/{selected_id}"))
        output_path = os.path.join(f"./outputs/merged_images/{selected_id}", f"merged_{i}.jpg")
        merged_image = cv2.cvtColor(merged_image, cv2.COLOR_RGB2BGR)
 

        avg_pixel_value = np.mean(merged_image)
        print(f'Average pixel value for {i}: {avg_pixel_value}')
        

        if avg_pixel_value < 255:
            object_list.append(i)
            images_for_video.append(merged_image)  
            cv2.imwrite(output_path, merged_image)
        
    output_video_path = f"./outputs/merged_video_{selected_id}.mp4" 
    for i in object_list:
        print(f'object_list: {i}')


    if images_for_video:

        height, width, _ = images_for_video[0].shape
        fourcc = cv2.VideoWriter_fourcc(*'mp4v') 
        out = cv2.VideoWriter(output_video_path, fourcc, 30.0, (width, height)) 

        for image in images_for_video:
            out.write(image) 

        out.release() 
        print(f"Video saved at {output_video_path}")
    else:
        print("No images to create a video.")
