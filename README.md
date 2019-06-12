# dcase2019_task1b
DCASE2019 Task 1b - Acoustic Scene Classification with mismatched recording devices 


## Task Description

For a detailed description of the task, data set, and baseline model, see:
TODO

## Setup

A step-by-step guide to train our model.

#### Environment
- download data set: https: TODO
- 'git clone https://github.com/OptimusPrimus/dcase2019_task1b.git' 
- place data into the data folder:
    ```
    .
    │
    └───data
    .   └───raw
    .       └───dcase20191b
                |   create_folds.py
                |   meta.csv 
    .           └───training
    .           .   |   tram-vienna-285-8639-a.wav
    .           .   |   ...
    .           └───test # leaderboard data
    .           .   |   1.wav
    .           .   |   ...
    .           └───audio # submission data
    .           .   |   1.wav
    .           .   |   ...
    .           └───evaluation_setup 
    .           .   |   fold1_evaluate.csv 
    .           .   |   fold1_test.csv
    .           .   |   fold1_train.csv
    .           └───evaluation_setup # empty
    |   ...
    
    ```

- import conda environment: `conda env create -f environment.yaml`
- install omniboard `npm install -g omniboard`

- [OPTIONAL] Add file `telegram.json` with Telegram Bot credentials to root: 
    ```json
    {
      "token": "SecretToken",
      "chat_id": "ChatID"
    }
    ```

#### Sacred

- add file `mongodb.json` with MongoDB credentials to root: 
    ```json
    {
      "user": "username",
      "pwd": "password123",
      "ip": "127.0.0.1",
      "db": "MIC",
      "port": 27017
    }
    ```

- run `python create_folds.py` in data/raw/dcase20191b


## Run an Experiment
- activate conda environment `source activate slap`
- edit `configs/dcase20191b.json`
- start training with `OMP_NUM_THREADS=1 CUDA_VISIBLE_DEVICES=0 python main.py with project=dcase20191b`

## Results Overview

Model performance measured in Accuray on devices B and C

| Method        |Sampling   | Augmentation   | AUROC         | Pre  | Rec  | F1  | ACC  |
|:-----------------:|:---------:|:--------------:|:-------------:|:---: |:---: |:---:|:---:|

*) CNN trained on Audio Set for feature extraction, classification with random forest.

