# 🏠 VR Room Reconstruction Easy step guidelines (Scanning to reconstruction is provided)


This project reconstructs a virtual room environment from video or multi-view images using segmentation and generation pipelines. The output is a structured 3D scene ready for editing or VR interaction.

---
## TO DO

- [x] Upload example dataset
- [ ] Link example dataset in README
- [ ] Draw pipeline diagram (Before meeting)
- [ ] Add module-level documentation
- [ ] Merge alignment code + simple viewer (optional)

## 📦 Features
- Easy i-phone ready to SLAM-based robust 3d reconstruction
- Object-level segmentation from RGB inputs
- Semantic and instance-aware labeling
- Supports Gaussian Splatting or mesh-based reconstruction

---




## 🚀 Pipeline Overview
![image](https://github.com/user-attachments/assets/64efd992-58d3-4b67-9123-c7b902b03a9e)


**Input:** Trained 3D Scene (Gaussian or Mesh), RGB images,

**Steps:** 
1. **Segmentation**  
   • Grounded-SAM / RAM (Optional - extract the tags -> for new examples)

   ![output 1](https://github.com/user-attachments/assets/977aa565-3102-4d6e-82dc-0b9c12bb4f28)

3. **Object Tracking**  
   • GroundingDino
4. **Object Reconstruction**  
   • Instance-level 3D mesh -> TRELLIS
5. **Mesh Alignment**  
   • Load .ply files into a mesh viewer or editor



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

Torch+cu118 installation

You can always install a different version than cuda 11.8 but should be aware that 11.8, 12.1, 12.4 are preferable.

```bash
pip install torch==2.4.0 torchvision==0.19.0 torchaudio==2.4.0 --index-url https://download.pytorch.org/whl/cu118
```






## Segmentation + Object tracking 

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



# Better results ( Use SAM - high quality )

Clone SAM_HQ2

```bash
git clone https://github.com/SysCV/sam-hq.git
cd sam-hq/sam-hq2
pip install -e .
```



Downloading checkpoints
```bash
wget https://huggingface.co/lkeab/hq-sam/resolve/main/sam2.1_hq_hiera_large.pt
mv sam2.1_hq_hiera_large.pt checkpoints
cd ../../
```

Downloading checkpoints directly ( place it in the sam-hq checkpoints folder)

<!-- - [sam2.1_hiera_large.pt](https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_large.pt) -->
- [sam2.1_hq_hiera_large.pt](https://huggingface.co/lkeab/hq-sam/resolve/main/sam2.1_hq_hiera_large.pt?download=true)

(note that these are the improved checkpoints denoted as SAM 2.1; see [Model Description](#model-description) for details.)


Update the file in miscs.py
```bash
mv misc.py sam-hq/sam-hq2/sam2/utils
```




# TRELLIS - object mesh generation
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


Update the example_multi_image in TRELLIS
```
mv example_multi_image.py TRELLIS
```


## Datasets (Example)
```bash
wget https://huggingface.co/datasets/johnkimryno/VR/resolve/main/images.zip
unzip images.zip
mkdir dataset
mv images dataset
rm -rf images.zip
```



# EXAMPLES

```bash
python tracking_with_dino.py
```

```bash
python white_background_with_specific_id_only_object.py
```

```bash
python TRELLIS/example_multi_image.py
```

