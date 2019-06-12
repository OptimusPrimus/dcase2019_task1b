# DCASE2019 Task 1b
DCASE2019 Task 1b - Acoustic Scene Classification with mismatched recording devices 


## Description
This repository contains a reproducible version of CP-JKU Student team's submission for [DCASE Challenge 2019](http://dcase.community/challenge2019/). A technical report describing this system will be available on the DCASE homepage as soon as official evaluation results are available. We need to stress that results might differ slightly from the ones described in the report, since we do not seed the random number generator manually. We therefore additionally provide saved model weights for all our submissions in `tmp/data/<model-id>` folder.

For a detailed description of task, data set, and baseline, see:
http://dcase.community/challenge2019/task-acoustic-scene-classification


## Step-by-step Guide for Prediction & Training

As a first step we need to set up the environment:
- Clone this project to you local machine:
    ```
    git clone https://github.com/OptimusPrimus/dcase2019_task1b.git
    ```
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

- Import conda environment & activate it: 
    ```
    conda env create -f environment.yml && source activate slap
    ```
- Create folds:
    ```
    cd data/raw/dcase20191b
    python create_folds.py
    ```
- Install Omniboard & MongoDB:
    
    ```
    npm install -g omniboard && conda install -c anaconda mongodb
    ```

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


### Predict

We then can use the provided model weights to creat predictions leaderboard and evaluation set.
The command to create predictions is:
```
python predict.py model_weights_file_1 [model_weights_file_2 model_weights_file_3 ...]
```

### [OPTIONAL] Train System Yourself

- Activate conda environment:
    ```
    source activate slap
    ```
- Edit `configs/dcase20191b.json`
- Train MSE, MI, and NoDA models with:
    ```
    OMP_NUM_THREADS=1 CUDA_VISIBLE_DEVICES=0 python main.py
    ```
    ```
    OMP_NUM_THREADS=1 CUDA_VISIBLE_DEVICES=0 python main.py with training.domain_daptation.class=domain_adaptation.MSE
    ```
    ```
    OMP_NUM_THREADS=1 CUDA_VISIBLE_DEVICES=0 python main.py with training.domain_daptation.class=domain_adaptation.MutualInformationDA
    ``
   
See [Sacred CLI Description](https://sacred.readthedocs.io/en/latest/command_line.html) on how to maipulate training parameters.
Logit outputs can be found in folder `data/tmp`
To see training progres, start Omniboard:
```
omniboard -m localhost:27017:MIC
``` 

## Citation

If you use the model or the model implementation please cite the following paper:
```
@inproceedings{Koutini2019Receptive,
    author      =   {Koutini, Khaled and Eghbal-zadeh, Hamid and Dorfer, Matthias and Widmer, Gerhard},
    title       =   {{The Receptive Field as a Regularizer in Deep Convolutional Neural Networks for Acoustic Scene Classification}},
    booktitle   =   {Proceedings of the European Signal Processing Conference (EUSIPCO)},
    address     =   {A Coru~{n}a, Spain},
    year        =   2019
}
```
If you use other parts of the implementation please cite:
```
@techreport{Primus2019DCASE,
    Author      =   {Primus, Paul and Eitelsebner, David},
    institution =   {{DCASE2018 Challenge}},
    title       =   {Acoustic Scene Classification with mismatched recording devices},
    month       =   {June},
    year        =   2019
}
```

## References


