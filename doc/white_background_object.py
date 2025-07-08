import os
import cv2
import numpy as np
import natsort

original_image_path = "./datasets/images/"
out_dir="./outputs_hq"
mask_data_dir = f"{out_dir}/mask_data/"
output_dir = f"{out_dir}/merged_images/"


if not os.path.exists(output_dir):
    os.makedirs(output_dir)

frame_num = len([f for f in os.listdir(original_image_path) if f.endswith(('.png', '.jpg', '.jpeg', '.PNG', '.JPG', '.JPEG'))])
full_image_list = os.listdir(original_image_path)

full_image_list = natsort.natsorted(full_image_list, key=lambda x: int(x.split('.')[0]))
full_name_list = [os.path.splitext(x)[0] for x in full_image_list]  

print(f'full_image_list: {full_image_list}')
print(f"Total frames found: {frame_num}")



json_data_dir = f"{out_dir}/json_data/"
target_class = "chair, table, wall, floor."
import os
import json
from collections import defaultdict
# 결과: {"frame_id1": [1, 3], "frame_id2": [5], ...}
class_instance_ids_per_frame = defaultdict(list)

for filename in os.listdir(json_data_dir):
    if filename.endswith(".json"):
        json_path = os.path.join(json_data_dir, filename)
        with open(json_path, "r") as f:
            data = json.load(f)
        
        frame_id = os.path.splitext(filename)[0].split("_")[-1]  # 예: "mask_29612235840" → "29612235840"
        labels = data.get("labels", {})

        for inst_info in labels.values():
            print(f'inst_info: {inst_info}')
            print(f'target_class: {target_class}')
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

        if not os.path.exists(os.path.join(f"{out_dir}/merged_images/{selected_id}")):
            os.makedirs(os.path.join(f"{out_dir}/merged_images/{selected_id}"))
        output_path = os.path.join(f"{out_dir}/merged_images/{selected_id}", f"merged_{int(i):06d}.jpg")
        merged_image = cv2.cvtColor(merged_image, cv2.COLOR_RGB2BGR)
 

        avg_pixel_value = np.mean(merged_image)
        print(f'Average pixel value for {i}: {avg_pixel_value}')
        

        if avg_pixel_value < 255:
            object_list.append(i)
            images_for_video.append(merged_image)  
            cv2.imwrite(output_path, merged_image)
        
    output_video_path = f"{out_dir}/merged_video_{selected_id}.mp4" 
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
