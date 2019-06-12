# dcase2019_task1b
DCASE2019 Task 1b - Acoustic Scene Classification with mismatched recording devices 


## Setup

For a detailed description of the task, data set, and baseline model, see:

TODO:


## Setup

- download data set: https: TODO
- place data into project:

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
#### Environment

- 'git clone https://github.com/OptimusPrimus/dcase2019_task1b.git' 
- import conda environment: `conda env create -f environment.yaml`
- add `export OMP_NUM_THREADS=1` at the end of .bashrc
- import conda environment `conda env create -f environment.yml`
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
  "pwd": "password",
  "ip": "127.0.0.1",
  "db": "MIC",
  "port": 27017
}
```

- run `python create_folds.py` in data/raw/dcase20191b


## Run an Experiment
- activate conda environment `source activate slap`
- edit `configs/dcase20191b.json`
- start training with `python main.py with project=dcase20191b`

## Results Overview

Model performance measured in Accuray on devices B and C

| Method        |Sampling   | Augmentation   | AUROC         | Pre  | Rec  | F1  | ACC  |
|:-----------------:|:---------:|:--------------:|:-------------:|:---: |:---: |:---:|:---:|

*) CNN trained on Audio Set for feature extraction, classification with random forest.

