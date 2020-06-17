# In Search of Robust Measures of Generalization

This repository contains the code, data, and analysis for the paper "In Search of Robust Measures of Generalization".

Data: 

## Directory structure
```
├── results
    └── tsvs
        └── ...
├── source
    ├── ...
    └── ...
├── experiments
    ├── coupled_network
    └── single_network
└── train.py
```

## Environment setup
### Conda
```bash
conda env create -f environment.yml
conda activate rgm
```

### Venv
```bash
sudo apt-get update
sudo apt-get upgrade
sudo apt-get install build-essential python3.8 python3.8-dev python3.8-distutils python3.8-venv
python3.8 -m venv env
source env/bin/activate
python3.8 -m pip install -U pip
python3.8 -m pip install -r requirements.txt
```
