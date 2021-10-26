## Problem Definition

- predict the probability of default for each user id in risk modeling
- default = 1 means defaulted users, default = 0 means otherwise
- Imbalance binary classification problem

## Expected Workflow

![title](img/WorkFlow.png)

## Variables (total = 43):

- uuid: text User Id <br>
- default: (or target) boolean (0, or 1) <br>
- Categorical, and numerical features are defined in `default_modeling.utils.preproc` (function `feature_definition`)

## Package Requirements:
- pandas, numpy, category_encoders, sklearn, scipy, joblib

## Folder Structure

![alt](img/tree_structure.png)

## DockerFile Contents

- My Local Working Directory named `/home/jupyter`. In this local folder:
  * `train_data` folder contains different files for training random forest classifers
  * `test_data` folder contains new data coming and waiting for prediction, prediction result will be locally stored inside the same file in this folder.
- Container will mount to those local folders: `train_data` and `test_data`
- With this approach, we can conveniently play with every new data coming, by replace the files inside `train_data` and/or `test_data`
```python
FROM python:3.8
WORKDIR /app/

RUN mkdir model

ENV TRAIN_FOLDER=./train_data
ENV TEST_FOLDER=./test_data
ENV TRAIN_FILE=train_set.csv
ENV TEST_FILE=test_set.csv
ENV MODEL_DIR=./model
ENV MODEL_NAME=risk_model
ENV TARGET=default

COPY requirements.txt .

RUN pip install -r requirements.txt
COPY default_modeling default_modeling

ENTRYPOINT ["python3"]
```
## Build Image from Dockerfile


```python
!docker build -t default_model -f Dockerfile .
```


## First, run unit test in Image to make sure everything's OK


```python
!docker run -t default_model:latest -m unittest discover default_modeling
```
    Found the following test data
    default_modeling/tests/data/test_sample_1.csv
    ..
    ----------------------------------------------------------------------
    Ran 2 tests in 0.772s
    
    OK


## Train with the first file `TRAIN_SET_1.csv` to obtain the first model


```python
!docker run -v /home/jupyter/train_data:/app/train_data default_model:latest \
                                                        -m default_modeling.train 
                                                        --train-file train_set_1.csv
```

    extracting arguments
    Training Data at ./train_data/train_set_1.csv
    Namespace(max_depth=10, min_samples_leaf=10, model_dir='./default_modeling/default_modeling/interface/', model_name='risk_model.joblib', n_estimators=100, random_state=1234, target='default', train_file='train_set_1.csv', trainingfolder='./train_data')
    Total Input Features 39
    class weight {0: 0.5071993428787708, 1: 35.22539149888143}
    Congratulation! Finish after 4.899581670761108 s


## Now replace by a new training file: TRAIN_SET_2.csv (the previous joblib model will be overwritten)


```python
!docker run -v /home/jupyter/train_data:/app/train_data default_model:latest \
                                                        -m default_modeling.train 
                                                        --train-file train_set_2.csv
```

    extracting arguments
    Training Data at ./train_data/train_set_2.csv
    Namespace(max_depth=10, min_samples_leaf=10, model_dir='./default_modeling/default_modeling/interface/', model_name='risk_model.joblib', n_estimators=100, random_state=1234, target='default', train_file='train_set_2.csv', trainingfolder='./train_data')
    Total Input Features 39
    class weight {0: 0.5074062934696794, 1: 34.255076142131976}
    Congratulation! Finish after 2.6793618202209473 s


## Now if some random forest hyperparameters needs to be modified:


```python
!docker run -v /home/jupyter/train_data:/app/train_data default_model:latest -m default_modeling.train \
                                                        --train-file train_set_2.csv \
                                                        --n-estimators 200 \
                                                        --max-depth 15 \
                                                        --min-samples-leaf 20
```

    extracting arguments
    Training Data at ./train_data/train_set_2.csv
    Namespace(max_depth=15, min_samples_leaf=20, model_dir='./default_modeling/default_modeling/interface/', model_name='risk_model.joblib', n_estimators=200, random_state=1234, target='default', train_file='train_set_2.csv', trainingfolder='./train_data')
    Total Input Features 39
    class weight {0: 0.5074062934696794, 1: 34.255076142131976}
    Congratulation! Finish after 3.2825217247009277 s


## Use image to predict new data 1 `test_set_1.csv`


```python
!docker run -v /home/jupyter/test_data:/app/test_data default_model:latest -m default_modeling.predict \
                                                      --test-file test_set_1.csv
```

    extracting arguments
    Model path: ./default_modeling/default_modeling/interface/risk_model.joblib
    Predicting test_set_1.csv ....
    Finish after 0.4421088695526123 s
    ...to csv ./test_data/test_set_1.csv


## Use image to predict new data 2 `test_set_2.csv`


```python
!docker run -v /home/jupyter/test_data:/app/test_data default_model:latest -m default_modeling.predict \
                                                      --test-file test_set_2.csv
```

    extracting arguments
    Model path: ./default_modeling/default_modeling/interface/risk_model.joblib
    Predicting test_set_2.csv ....
    Finish after 0.23205804824829102 s
    ...to csv ./test_data/test_set_2.csv



## We have prediction in local folder test_data. Evaluate with Metrics

- Decision threshold on the probability of default would probably depend on credit policy. There could be several cutoff points or a mathematical cost function rather than a fixed decision threshold. Therefore, binary metrics like F1, Recall, or Precision is not meaningful in this situation. And the output should be a prediction in probability.
- KS-statistic (between P(prediction|truth = 1) and P(prediction|truth = 0) to quantify the distance between 2 classes) are used to evaluate model.
- Left plot: ROC AUC Curve
- Right plot: Normalized KS Distribution of 2 types of users:
  * class 0: non-default
  * class 1: default

![alt](img/AUC.png) ![alt](img/KS_Curve.png)

## Conclusions & Future Work

- With KS score = 0.66 and small p-value, this means the predictor can properly distinguish between default and non-default users (test is significant)
- Visually, we can observe the clear gap in the KS distribution plot between 2 classes
- In the future, host with AWS Sagemeker endpoint
