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
python3 -m venv env
source env/bin/activate
python -m pip install -U pip
python -m pip install -r requirements.txt
```
