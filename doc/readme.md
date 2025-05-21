# 🏠 VR Room Reconstruction Project

This project reconstructs a virtual room environment from video or multi-view images using segmentation and generation pipelines. The output is a structured 3D scene ready for editing or VR interaction.

---

## 📦 Features

- Object-level segmentation from RGB inputs
- Semantic and instance-aware labeling
- 3D room layout generation using lifted masks
- Supports Gaussian Splatting or mesh-based reconstruction
- Editable scene graph for interactive VR use

---




## 🚀 Pipeline Overview

**Input:** Trained 3D Scene (Gaussian or Mesh), RGB images,

**Steps:**
1. **Segmentation**  
   • Grounded-SAM / RAM
2. **Object Tracking**  
   • Across views or time
3. **Object Reconstruction**  
   • Instance-level 3D mesh / Gaussian group
4. **Mesh Alignment**  
   • Top-down layout or VR axis alignment



## 🔧 Setup

Python version : 3.10.16
GPU : RTX 4090
CUDA : 11.8
Ubuntu : 24.04

```bash
conda create -n seggen python==3.10.16 -y
conda activate seggen
mkdir SegGen
cd SegGen
```

## Torch+cu118 installation

```bash
pip install torch==2.4.0 torchvision==0.19.0 torchaudio==2.4.0 --index-url https://download.pytorch.org/whl/cu118
```

Git clone Grounded SAM 2
```bash
git clone https://github.com/IDEA-Research/Grounded-SAM-2.git
cd Grounded-SAM-2
```

Install `Segment Anything 2`:

```bash
pip install -e .
```

Install `Grounding DINO`:

```bash
pip install --no-build-isolation -e grounding_dino
```
Download the checkpoints
```bash
sh gdino_checkpoints/download_ckpts.sh
mv groundingdino_swinb_cogcoor.pth gdino_checkpoints/
mv groundingdino_swint_ogc.pth gdino_checkpoints/
cd ../
```





Clone SAM_HQ2

```bash
git clone https://github.com/SysCV/sam-hq.git
cd sam-hq/sam-hq2
pip install -e .
```

Download the checkpoints individually from:


<!-- - [sam2.1_hiera_large.pt](https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_large.pt) -->
- [sam2.1_hq_hiera_large.pt](https://huggingface.co/lkeab/hq-sam/resolve/main/sam2.1_hq_hiera_large.pt?download=true)

(note that these are the improved checkpoints denoted as SAM 2.1; see [Model Description](#model-description) for details.)

or

```bash
wget https://huggingface.co/lkeab/hq-sam/resolve/main/sam2.1_hq_hiera_large.pt
mv sam2.1_hq_hiera_large.pt checkpoints
cd ../../
```




### TRELLIS
```bash
git clone --recurse-submodules https://github.com/microsoft/TRELLIS.git
cd TRELLIS
cd trellis/representations/mesh/
rm -rf flexicubes
git clone https://github.com/MaxtirError/FlexiCubes.git
mv FlexiCubes flexicubes
cd ../../../
```

```sh
. ./setup.sh --basic --diffoctreerast --spconv --mipgaussian --nvdiffrast
pip install xformers==0.0.27.post2 --index-url https://download.pytorch.org/whl/cu118
pip install kaolin -f https://nvidia-kaolin.s3.us-east-2.amazonaws.com/torch-2.4.0_cu118.html  #These are just lines since the shell script has issues
cd ../
```
#skip flash attention


```bash
python 
```


```bash
python TRELLIS/example_multi_image.py
```

## TO DO
- mv example_multi_image to mine
- Upload example dataset
- Link example dataset
- 
