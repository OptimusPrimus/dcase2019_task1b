# dcase2019_task1b
DCASE2019 Task 1b - Acoustic Scene Classification with mismatched recording devices 


## Task Description

For a detailed description of the task, data set, and baseline model, see:
http://dcase.community/challenge2019/task-acoustic-scene-classification

## Prepare

A step-by-step guide to train our model.

#### Environment

- Clone this project: `git clone https://github.com/OptimusPrimus/dcase2019_task1b.git`
- download and place data set into the data folder:
    ```
    .
    │
    └───data
    .   └───raw
    .       └───dcase20191b
    .           |   create_folds.py
    .           |   meta.csv 
    .           └───training                            # development data
    .           .   |   tram-vienna-285-8639-a.wav
    .           .   |   ...
    .           └───test                                # leaderboard data
    .           .   |   1.wav
    .           .   |   ...
    .           └───audio                               # submission data
    .           .   |   1.wav
    .           .   |   ...
    .           └───evaluation_setup                    # evaluation split
    .           .   |   fold1_evaluate.csv 
    .           .   |   fold1_test.csv
    .           .   |   fold1_train.csv
    .           └───training_setup                      # empty, 4-flod CV split
    |   ...
    
    ```

- import conda environment: `conda env create -f environment.yaml`
- run `python create_folds.py` in data/raw/dcase20191b
- install omniboard `npm install -g omniboard`

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
- [OPTIONAL] Add file `telegram.json` with Telegram Bot credentials to root: 
    ```json
    {
      "token": "SecretToken",
      "chat_id": "ChatID"
    }
    ```

## Run Experiment
- activate conda environment `source activate slap`
- edit `configs/dcase20191b.json`
- train with `OMP_NUM_THREADS=1 CUDA_VISIBLE_DEVICES=0 python main.py`

## Restults

- Start Omniboard with `omniboard -m localhost:27017:MIC` to see training progress.
- Logit outputs can be found in folder `data/tmp`

## Predict

TODO