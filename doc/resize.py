import os
from PIL import Image

input_dir = "datasets/images_john"
output_dir = "datasets/images_john_resized"

os.makedirs(output_dir, exist_ok=True)

for fname in os.listdir(input_dir):
    if fname.lower().endswith(('.png', '.jpg', '.jpeg')):
        img_path = os.path.join(input_dir, fname)
        img = Image.open(img_path)
        w, h = img.size
        img_resized = img.resize((w // 2, h // 2), Image.Resampling.LANCZOS)
        img_resized.save(os.path.join(output_dir, fname))
        print(f"{fname}: {w}x{h} -> {w//2}x{h//2}")
