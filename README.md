# dcase2019_task1b
DCASE2019 Task 1b - Acoustic Scene Classification with mismatched recording devices 


## Setup

For a detailed description of the data set and the Baseline Model*, see:
TODO

## Setup

- download data set: https://bmcfee.github.io/papers/ismir2018_openmic.pdf
- place data into project:
```
.
│
└───data
.   └───raw
.       └───openMIC
.           |   openmic-2018.npz
.           |   prepare.py
.           └───files
.           .   |   000046_3840.ogg
.           .   |   ...
.           └───partitions
.               |   split01_test.csv
.               |   split01_train.csv
|   ...

```
- run `python data/raw/openMIC/prepare.py`
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

- add `export OMP_NUM_THREADS=1` at the end of .bashrc

- import conda environment: `conda env create -f environment.yaml`

- [OPTIONAL] Add file `telegram.json` with Telegram Bot credentials to root: 
```json
{
  "token": "SecretToken",
  "chat_id": "ChatID"
}
```


## Run an Experiment
- activate conda environment `source activate slap`
- edit `configs/openmic.json`
- start training with `python main.py with project=openmic`

## Results Overview

Model performance measured in hmeaned AUROC, Precission, Recall, F1-Score over 20 classes.

| Model Name        |Sampling   | Augmentation   | AUROC         | Pre  | Rec  | F1  | ACC  |
|:-----------------:|:---------:|:--------------:|:-------------:|:---: |:---: |:---:|:---:|
| CP_ResNet  	    | random       |     yes        |	.8984   | ?     | ?    | ?   | ?   |
| MIC Baseline*  | no        |     no         | .88??		     | ?    | ?    | ?   | ?   |
| CP_ResNet  	    | random       |     no        |	.8878   		     | .6430    | .8458    | .7306  | .7853   |
| Baseline  	    | random       |     yes        |	.8439   		     | .6170    | .7754    | .6866  | .7625   |
| Baseline  	    | rand over       |     yes        |	.8424		     | .6047    | .7808    | .6812  | .7562   |

*) CNN trained on Audio Set for feature extraction, classification with random forest.

