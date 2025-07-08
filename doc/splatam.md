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
unzip experiments
rm experiments.zip
