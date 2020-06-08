# In Search of Robust Measures of Generalization

This repository contains the code, data, and analysis for the paper "In Search of Robust Measures of Generalization".

## Environment setup
### Conda
```bash
conda env create -f environment.yml
conda activate rgm
```

### Venv
```bash
sudo apt-get install python3.8 python3.8-dev python3.8-distutils python3.8-venv
python3.8 -m venv env
source env/bin/activate
python3.8 -m pip install -U pip
python3.8 -m pip install -r requirements.txt
```

## Directory structure
```
├── results
    └── tsvs
        └── ...
├── scripts
    ├── ...
    └── ...
├── source
    ├── ...
    └── ...
└── run.py
```
