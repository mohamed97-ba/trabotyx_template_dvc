# trabotyx_template_dvc

Directory Structure
--------------------

    .
    ├── README.md
    ├── stabber
        ├── efficient_net <- Pretrained Model
        ├── dataloader.py 
        ├── network.py <- Model Architecture 
        ├── train.py <- train model stage code
        ├── predict.py <- predict model stage code
        └── utils <- auxiliary functions and classes (data preprocessing)
    ├── dvcfiles
        ├── data.dvc  
        └── trained_model.dvc
    ├── data
        └── test/
        │
        ├── train/
        │   ├── n01440764/
        │   ├── n02102040/
        │   ├── n02979186/
        │   ├── n03000684/
        │   ├── n03028079/
        │   ├── n03394916/
        │   ├── n03417042/
        │   ├── n03425413/
        │   ├── n03445777/
        │   └── n03888257/
        |
        └── val/
            ├── n01440764/
            ├── n02102040/
            ├── n02979186/
            ├── n03000684/
            ├── n03028079/
            ├── n03394916/
            ├── n03417042/
            ├── n03425413/
            ├── n03445777/
            └── n03888257/

# Preparation

### 1. Clone this repository

```bash
git clone https://github.com/trabotyx/trabotyx_template_dvc.git
```

cd trabotyx_template_dvc 

### 2. Get data

Download data from Amazon s3

```bash
wget '''s3_uri_path_to_data'''
```         
###  Create a branch for you first experiment
```bash
git checkout -b "first experiment"
```  
### 3. Initialize DVC init 

__1) Install DVC__ 
`pip install dvc`

[Link for installation instructions](https://dvc.org/doc/get-started/install)

__2) Initialize DVC init__
ONLY if you build the project from scratch. For projects clonned from GitHub it's already initialized.

Initialize DVC 
```bash
dvc init
```

Commit dvc init

```bash
git commit -m "Initialize DVC"
``` 

__3) Add remote storage for DVC (any local folder)__
```bash

dvc remote add -d storage s3://mybucket/dvcstore
git add .dvc/config
git commit -m "Configure remote storage"
```
# Tracking the data

- Now let's navigate to the dvcfiles

```bash
dvc add ../data --file data.dvc
``` 

The command will generate 2 files: .gitignore and .dvc
* .gitignore — This file excludes a file/folder from a Git repository.
* data.dvc — This file is metadata storing information about the added file/folder and associates with a specific version of data.

To version control our data, this file needs to be added into a Git repository using the following commands:

```bash
git add .gitignore data.dvc
git commit -m 'Added dataset (version) v..'
``` 

To upload your files from the cache to the remote, use the push command:

```bash
dvc push
``` 

Your data is now safely stored in a location away from your repository. Finally, push the files under Git control to GitHub:

```bash
git push --set-upstream origin first_experiment
``` 

## Versioning the data

```bash
git tag -a "v0.0" -m "dataset Version 0.0"
``` 
```bash
git push origin v0.0
``` 
## Pipeline stages

```bash
dvc run -n train  \
-p  train.epochs,data.batch_size \
-d /home/med-ba/trabotyx_template_dvc/stabber/train.py  \
-o /home/med-ba/trabotyx_template_dvc/stabber/result/output/model_best.pth \
-M metric.json 
python /home/med-ba/trabotyx_template_dvc/stabber/train.py  --param /home/med-ba/trabotyx_template_dvc/params.yaml 
``` 
A dvc.yaml file is generated. It includes information about the command we want to run (python /home/med-ba/trabotyx_template_dvc/stabber/train.py  --param /home/med-ba/trabotyx_template_dvc/params.yaml ), its dependencies, and outputs.
- dvc.yaml
```bash
stages:
  train:
    cmd: python /home/med-ba/trabotyx_template_dvc/stabber/train.py --config /home/med-ba/trabotyx_template_dvc/stabber/config.yml
    deps:
    - stabber/config.yml
    - stabber/train.py
    outs:
    - stabber/result/output/model_best.pth
    metrics:
    - stabber/result/output/log.json:
        cache: false
``` 

** we can add manually different stage:
```bash
stages:
  train:
    cmd: python /home/med-ba/trabotyx_template_dvc/stabber/train.py --param /home/med-ba/trabotyx_template_dvc/params.yaml
    deps:
    - stabber/train.py
    params:
    - data.batch_size
    - train.epochs
    outs:
    - stabber/result/output/model_best.pth
    metrics:
    - metric.json:
        cache: false
  predict:
    cmd: python /home/med-ba/trabotyx_template_dvc/stabber/predict.py --config /home/med-ba/trabotyx_template_dvc/stabber/config.yml
    deps:
    - stabber/predict.py
    outs:
    - stabber/result/visualise_stab_bboxtn
```
and run : 
```bash
dvc repro
``` 
to reproduce the pipeline

## Data registy
migrate to dvc 2.0.5
Adding datasets to a registry can be as simple as placing the data file or directory in question inside the workspace, and track it with dvc add.
## Data downloads
```bash
dvc get https://github.com/trabotyx/trabotyx_dvc_stab_nostab.git data
``` 
This downloads /data from the project trabotyx_dvc_stab_nostab and places it in the current working directory.




