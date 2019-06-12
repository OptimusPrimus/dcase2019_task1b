# DCASE2019 Task 1b
DCASE2019 Task 1b - Acoustic Scene Classification with mismatched recording devices 


## Task Description

For a detailed description of the task, data set, and baseline model, see:
http://dcase.community/challenge2019/task-acoustic-scene-classification

## Prepare

Step-by-step guide to train our model:

#### Environment

- Clone this project to you local machine:
    ```git clone https://github.com/OptimusPrimus/dcase2019_task1b.git```
- Download and place data set into the data folder:
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

- Import conda environment: `conda env create -f environment.yaml`
- In data/raw/dcase20191b, run
    ```python create_folds.py```
- Install omniboard:
    ```npm install -g omniboard```

#### Sacred

- Add file `mongodb.json` with MongoDB credentials to root: 
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
- Activate conda environment:
```source activate slap```
- Edit `configs/dcase20191b.json`
- Train MSE, MI, and NoDA models with:
    ```OMP_NUM_THREADS=1 CUDA_VISIBLE_DEVICES=0 python main.py``
    ```OMP_NUM_THREADS=1 CUDA_VISIBLE_DEVICES=0 python main.py with training.domain_daptation.class=domain_adaptation.MSE``
    ```OMP_NUM_THREADS=1 CUDA_VISIBLE_DEVICES=0 python main.py with training.domain_daptation.class=domain_adaptation.MutualInformationDA``

## Restults

- To see training progres, start Omniboard: 
```omniboard -m localhost:27017:MIC``` 
- Logit outputs can be found in folder `data/tmp`

## Predict

TODO
