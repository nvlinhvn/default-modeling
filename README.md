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

- My Local Working Directory named `/home/jupyter`. In this local working directory:
  * `train_data` folder contains different files for training random forest classifers
  * `model` folder store the trained `.joblib` random forest, and the model will be loaded in this folder for prediction
  * `test_data` folder contains new data coming and waiting for prediction, prediction result will be locally stored inside the same file in this folder.
- Container will mount to those local folders: `train_data`, `test_data` and `model`
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
!docker run -v /home/jupyter/train_data:/app/train_data \
            -v /home/jupyter/model:/app/model \
            default_model:latest -m default_modeling.default_modeling.interface.train \
            --train-file train_set_1.csv
```

    extracting arguments
    Namespace(max_depth=10, min_samples_leaf=10, model_dir='./model', model_name='risk_model', n_estimators=100, random_state=1234, target='default', train_file='train_set_1.csv', train_folder='./train_data')
    Training Data at ./train_data/train_set_1.csv
    Total Input Features 39
    class weight {0: 0.5071993428787708, 1: 35.22539149888143}
    Congratulation! Saving model at ./model/risk_model.joblib. Finish after 5.14886212348938 s


## Now replace by a new training file: TRAIN_SET_2.csv (the previous joblib model will be overwritten)


```python
!docker run -v /home/jupyter/train_data:/app/train_data \
            -v /home/jupyter/model:/app/model \
            default_model:latest -m default_modeling.default_modeling.interface.train \
            --train-file train_set_2.csv
```

    extracting arguments
    Namespace(max_depth=10, min_samples_leaf=10, model_dir='./model', model_name='risk_model', n_estimators=100, random_state=1234, target='default', train_file='train_set_2.csv', train_folder='./train_data')
    Training Data at ./train_data/train_set_2.csv
    Total Input Features 39
    class weight {0: 0.5074062934696794, 1: 34.255076142131976}
    Found existing model at: ./model/risk_model.joblib.
    Overwriting ...
    Congratulation! Saving model at ./model/risk_model.joblib. Finish after 2.718583345413208 s


## Now if some random forest hyperparameters needs to be modified:


```python
!docker run -v /home/jupyter/train_data:/app/train_data \
            -v /home/jupyter/model:/app/model \
            default_model:latest -m default_modeling.default_modeling.interface.train \
            --train-file train_set_2.csv \
            --n-estimators 200 \
            --max-depth 15 \
            --min-samples-leaf 20
```

    extracting arguments
    Namespace(max_depth=15, min_samples_leaf=20, model_dir='./model', model_name='risk_model', n_estimators=200, random_state=1234, target='default', train_file='train_set_2.csv', train_folder='./train_data')
    Training Data at ./train_data/train_set_2.csv
    Total Input Features 39
    class weight {0: 0.5074062934696794, 1: 34.255076142131976}
    Found existing model at: ./model/risk_model.joblib.
    Overwriting ...
    Congratulation! Saving model at ./model/risk_model.joblib. Finish after 3.4293792247772217 s


## Use image to predict new data 1 `test_set_1.csv`


```python
!docker run -v /home/jupyter/test_data:/app/test_data  \
            -v /home/jupyter/model:/app/model default_model:latest \
            -m default_modeling.default_modeling.interface.predict \
            --test-file test_set_1.csv         
```

    extracting arguments
    Namespace(model_dir='./model', model_name='risk_model', target='default', test_file='test_set_1.csv', test_folder='./test_data')
    Found model at: ./model/risk_model.joblib
    Predicting test_set_1.csv ....
    Finish after 0.5522034168243408 s
    ...to csv ./test_data/test_set_1.csv


## Use image to predict new data 2 `test_set_2.csv`


```python
!docker run -v /home/jupyter/test_data:/app/test_data  \
            -v /home/jupyter/model:/app/model default_model:latest \
            -m default_modeling.default_modeling.interface.predict \
            --test-file test_set_2.csv                                                 
```

    extracting arguments
    Namespace(model_dir='./model', model_name='risk_model', target='default', test_file='test_set_2.csv', test_folder='./test_data')
    Found model at: ./model/risk_model.joblib
    Predicting test_set_2.csv ....
    Finish after 0.3289515972137451 s
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
