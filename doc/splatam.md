# Splatam -> reconstruction pipeline to alignment


## ✅ Step-by-Step Installation

### 1. Clone the SplaTAM repository
```bash
#Start at SegGen
wget https://huggingface.co/datasets/johnkimryno/VR/resolve/main/SplaTAM_.zip
unzip SplaTAM_.zip
rm SplaTAM_.zip
mv SplaTAM_ SplaTAM
cd SplaTAM

git clone git@github.com:git@github.com:JonathonLuiten/diff-gaussian-rasterization-w-depth.git
cd diff-gaussian-rasterization-w-depth
python setup.py install
pip install .

cd ../

pip install -r requirements.txt
```

### 2. Download the data
```bash
wget https://huggingface.co/datasets/johnkimryno/VR/resolve/main/experiments.zip
#UNZIP_DISABLE_ZIPBOMB_DETECTION=TRUE unzip
unzip experiments
rm experiments.zip
```

### 3. Final Evaluation
```bash
python scripts/final_eval.py
```

### 4. Label gs by voting
```bash
cp ../../Room_recon/doc/label_gs_by_voting.py utils/
python utils/label_gs_by_voting.py
```

### 5. Floor normal
```bash
cp ../../Room_recon/doc/floor_normal.py utils/
python utils/floor_normal.py
```

### 6. Hough line detector
```bash
python hough_line_detector.py
```

### 7. Back project
```bash
python project_back.py
```

### 8. Load the cad models and reorient
```bash
cp ../../Room_recon/doc/load_models.py scripts/
python scripts/load_models.py
```
