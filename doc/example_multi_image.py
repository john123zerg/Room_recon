import os
os.environ['ATTN_BACKEND'] = 'xformers'   # Can be 'flash-attn' or 'xformers', default is 'flash-attn'
os.environ['SPCONV_ALGO'] = 'native'        # Can be 'native' or 'auto', default is 'auto'.
                                            # 'auto' is faster but will do benchmarking at the beginning.
                                            # Recommended to set to 'native' if run only once.

import numpy as np
import imageio
import random
import re
import glob
from PIL import Image
from trellis.pipelines import TrellisImageTo3DPipeline
from trellis.utils import render_utils
from natsort import natsorted
from trellis.pipelines import TrellisImageTo3DPipeline
from trellis.utils import render_utils, postprocessing_utils

# Load a pipeline from a model folder or a Hugging Face model hub.
pipeline = TrellisImageTo3DPipeline.from_pretrained("gqk/TRELLIS-image-large-fork")
pipeline.cuda()

# 이미지 디렉토리와 비디오 디렉토리 설정
input_number='1'
image_dir = f"../Seggen/results_free/merged_images/{input_number}"
outputs_dir = "outputs_Jul_2"
os.makedirs(outputs_dir, exist_ok=True)
os.makedirs(os.path.join(outputs_dir, "result"), exist_ok=True)

# merged_video 파일들 찾기
# merged_video_files = glob.glob(os.path.join(outputs_dir, "merged_video_*.mp4"))
# if not merged_video_files:
#     print("No merged_video files found. Using default.")
merged_video_files = [os.path.join(outputs_dir, f"merged_video_{input_number}.mp4")]
    
print(f"Found {len(merged_video_files)} merged_video files: {merged_video_files}")

valid_exts = [".png", ".jpg", ".jpeg"]

# 이미지 파일 경로 리스트 (자연 정렬)
image_paths = natsorted([
    os.path.join(image_dir, fname)
    for fname in os.listdir(image_dir)
    if os.path.splitext(fname)[-1].lower() in valid_exts
])

# 이미지 불러오기
images = [Image.open(p) for p in image_paths]

def add_white_background(frames, bg_color=(255, 255, 255)):
    new_frames = []
    for frame in frames:
        h, w, c = frame.shape
        bg = np.ones_like(frame) * np.array(bg_color, dtype=np.uint8)  # 흰색 배경
        alpha = (frame[..., :3] != 0).any(axis=2).astype(np.uint8)[..., None]  # 배경 아닌 부분 마스크
        blended = bg * (1 - alpha) + frame * alpha
        new_frames.append(blended.astype(np.uint8))
    return new_frames

# 각 merged_video 파일에 대해 처리
for merged_video_path in merged_video_files:
    # 비디오 ID 추출
    match = re.search(r"merged_video_(\d+)\.mp4", os.path.basename(merged_video_path))
    video_id = match.group(1) if match else "unknown"
    print(f"Processing video ID: {video_id}")
    
    # 각 시드의 결과를 저장할 리스트
    seed_results = []
    
    # 각 비디오당 4번 다른 시드로 실행
    for run_idx in range(4):
        # 랜덤 시드 생성
        random_seed = random.randint(1, 10000)
        print(f"Run #{run_idx+1} with seed: {random_seed}")
        
        # Run the pipeline with random seed
        outputs = pipeline.run_multi_image(
            images,
            seed=random_seed,
            
            # Optional parameters
            sparse_structure_sampler_params={
                "steps": 12,
                "cfg_strength": 7.5,
            },
            slat_sampler_params={
                "steps": 12,
                "cfg_strength": 3,
            },
        )
        
        # 시드별 결과 저장 (GS와 mesh)
        video_gs = render_utils.render_video(outputs['gaussian'][0])['color']
        video_gs_white = add_white_background(video_gs)
        
        video_mesh = render_utils.render_video(outputs['mesh'][0])['normal']
        
        seed_results.append({
            'seed': random_seed,
            'gs': video_gs_white,
            'mesh': video_mesh,
            'gaussian': outputs['gaussian'][0],  # 3D Gaussian 객체 저장
            'mesh_obj': outputs['mesh'][0]       # Mesh 객체 저장
        })
            # 각 시드별로 GLB와 PLY 파일 저장
        seed_output_dir = os.path.join(outputs_dir, f"seed_outputs_{video_id}")
        os.makedirs(seed_output_dir, exist_ok=True)
        
        # GLB 파일 생성 및 저장
        glb = postprocessing_utils.to_glb(
            outputs['gaussian'][0],
            outputs['mesh'][0],
            # Optional parameters
            simplify=0.95,          # Ratio of triangles to remove in the simplification process
            texture_size=1024,      # Size of the texture used for the GLB
        )
        glb_filename = f"output_{video_id}_seed_{random_seed}.glb"
        glb_path = os.path.join(seed_output_dir, glb_filename)
        glb.export(glb_path)
        print(f"Saved GLB file: {glb_path}")
        
        # PLY 파일 저장 (Gaussians)
        ply_filename = f"gaussians_{video_id}_seed_{random_seed}.ply"
        ply_path = os.path.join(seed_output_dir, ply_filename)
        outputs['gaussian'][0].save_ply(ply_path)
        print(f"Saved PLY file: {ply_path}")
    # 모든 시드의 결과가 준비되면 비디오 생성
    
    # 프레임 크기 정보 (모든 결과는 동일한 크기임)
    frame_h, frame_w, _ = seed_results[0]['gs'][0].shape
    
    # 입력 이미지 준비
    input_img = images[0].resize((frame_w, frame_h // 2)).convert("RGB")
    input_np = np.array(input_img)
    
    # 해당 video_id에 맞는 merged 폴더에서 첫 번째 이미지 찾기
    merged_folder = os.path.join(outputs_dir, f"merged_{video_id}")
    result_img = None
    
    if os.path.exists(merged_folder):
        # 해당 폴더 내 모든 이미지 파일 찾기
        merged_images = []
        for ext in valid_exts:
            merged_images.extend(glob.glob(os.path.join(merged_folder, f"*{ext}")))
        
        # 이미지 파일이 있으면 첫 번째 이미지 사용
        if merged_images:
            merged_images = natsorted(merged_images)
            result_img = Image.open(merged_images[0])
            print(f"Using first image from merged_{video_id} folder: {merged_images[0]}")
    
    # 이미지를 찾지 못했으면 이전 방식대로 result 폴더에서 찾기
    if result_img is None:
        result_path = os.path.join(outputs_dir, "result", f"result_{video_id}.jpg")
        if not os.path.exists(result_path):
            result_path = os.path.join(outputs_dir, "result", f"result_{video_id}.png")
        
        if os.path.exists(result_path):
            result_img = Image.open(result_path)
            print(f"Using result image: {result_path}")
        else:
            result_img = Image.new("RGB", (frame_w, frame_h // 2), (255, 255, 255))
            print(f"No result image found for ID {video_id}, using blank image")
    
    result_img = result_img.resize((frame_w, frame_h // 2)).convert("RGB")
    result_np = np.array(result_img)
    
    # 최종 비디오 프레임 생성
    video_final = []
    
    # 각 프레임마다 2/2/2/2/2 형식으로 배치
    for frame_idx in range(len(seed_results[0]['gs'])):
        # 첫째 줄: 입력 이미지와 merged 결과 이미지
        top_row = np.concatenate([input_np, result_np], axis=1)
        
        # 각 시드별 결과를 한 행씩 추가 (GS와 mesh를 가로로 배치)
        rows = [top_row]
        for result in seed_results:
            gs_frame = result['gs'][frame_idx]
            mesh_frame = result['mesh'][frame_idx]
            seed_row = np.concatenate([gs_frame, mesh_frame], axis=1)
            rows.append(seed_row)
        
        # 모든 행을 세로로 합치기
        full_frame = np.vstack(rows)
        video_final.append(full_frame.astype(np.uint8))
    
    # 비디오 저장
    output_video_path = os.path.join(outputs_dir, f"output_{video_id}_all_seeds.mp4")
    imageio.mimsave(output_video_path, video_final, fps=30)
    print(f"Saved combined output to: {output_video_path}")
    
    
