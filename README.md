# Monorepo with Modes dependencies

# Demo Project Structure
------------------------
```
    .
    ├── data
    │   ├── processed               <- processed data
    │   └── raw                     <- original unmodified/raw data
    ├── models                      <- folder for ML models
    ├── notebooks                   <- Jupyter Notebooks (ingored by Git)
    ├── reports                     <- folder for experiment reports
    ├── model-a                     <- Model A sub-directory (DVC repo)
    ├── model-b                     <- Model B sub-directory (DVC repo)
    ├── model-c                     <- Model C sub-directory (DVC repo)
    └── README.md
```

## About 

- models A and B outputs are used by Model C as inputs 


### Create a Virtual environment

Create virtual environment named `.venv` (you may use other name)
```bash
python3 -m venv .venv
echo "export PYTHONPATH=$PWD" >> .venv/bin/activate
source .venv/bin/activate
pip install --upgrade pip setuptools wheel
pip install -r requirements.txt
```


## Setup projects 

### Init main DVC project (in root dir)
```bash
    # Navigate to `model_a` dir: cd model_a
    dvc init 
    git add .dvc/config .dvc/.gitignore && git commit -m "Initialize DVC project"
``` 

### Setup `Model A` project 
```bash
    # Navigate to `model_a` dir: cd model_a
    dvc init --subdir
    git add .dvc/config .dvc/.gitignore && git commit -m "Initialize DVC project A"
``` 

### Setup `Model B` project 
```bash
    # Navigate to `model_b` dir: cd model_b
    dvc init --subdir
    git add .dvc/config .dvc/.gitignore && git commit -m "Initialize DVC project B"
``` 

### Setup `Model C` project 
```bash
    # Navigate to `model_c` dir: cd model_c
    dvc init --subdir
    git add .dvc/config .dvc/.gitignore && git commit -m "Initialize DVC project C"
``` 


## Experimenting with Monorepo 

```bash
    dvc exp run
``` 
