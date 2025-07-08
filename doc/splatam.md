# Splatam -> reconstruction pipeline to alignment


## ✅ Step-by-Step Installation

### 1. Clone the SplaTAM repository
```bash
#Start at SegGen
wget https://huggingface.co/datasets/johnkimryno/VR/resolve/main/SplaTAM%202.zip
mv 'SplaTAM 2' SplaTAM.zip
unzip SplaTAM%202
cd SplaTAM

git clone git@github.com:git@github.com:JonathonLuiten/diff-gaussian-rasterization-w-depth.git
cd diff-gaussian-rasterization-w-depth
python setup.py install
pip install .

cd ../

pip install -r requirements.txt
```
