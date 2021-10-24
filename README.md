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

ENV TRAINING_FOLDER=./train_data
ENV TESTING_FOLDER=./test_data
ENV TRAIN_FILE=train_set.csv
ENV TEST_FILE=test_set.csv
ENV MODEL_DIR=./default_modeling/default_modeling/interface/
ENV MODEL_NAME=risk_model.joblib
ENV TARGET=default

COPY requirements.txt .

RUN pip install -r requirements.txt
COPY default_modeling default_modeling

ENTRYPOINT ["python3"]
```
## Build Image from Dockerfile


```python
!docker build --no-cache -t default_model -f Dockerfile .
```
    Sending build context to Docker daemon  87.86MB
    Step 1/13 : FROM python:3.8
     ---> 79372a158581
    Step 2/13 : WORKDIR /app/
     ---> Running in 6530deb397a3
    Removing intermediate container 6530deb397a3
     ---> 9f7c50fb88f3
    Step 3/13 : ENV TRAINING_FOLDER=./train_data
     ---> Running in d55ad35cf081
    Removing intermediate container d55ad35cf081
     ---> 869c519c5bb0
    Step 4/13 : ENV TESTING_FOLDER=./test_data
     ---> Running in 7a1789babd03
    Removing intermediate container 7a1789babd03
     ---> 006a1ce528d9
    Step 5/13 : ENV TRAIN_FILE=train_set.csv
     ---> Running in 808a93a68878
    Removing intermediate container 808a93a68878
     ---> c73a4850150d
    Step 6/13 : ENV TEST_FILE=test_set.csv
     ---> Running in 2d846111b869
    Removing intermediate container 2d846111b869
     ---> 6c58e5495477
    Step 7/13 : ENV MODEL_DIR=./default_modeling/default_modeling/interface/
     ---> Running in 0f645129bedc
    Removing intermediate container 0f645129bedc
     ---> dc0107708c22
    Step 8/13 : ENV MODEL_NAME=risk_model.joblib
     ---> Running in aca69b247043
    Removing intermediate container aca69b247043
     ---> 6c08eaea4391
    Step 9/13 : ENV TARGET=default
     ---> Running in 1e6cd082e31e
    Removing intermediate container 1e6cd082e31e
     ---> e119b3e769ee
    Step 10/13 : COPY requirements.txt .
     ---> 39eeced1ef52
    Step 11/13 : RUN pip install -r requirements.txt
     ---> Running in 5c63f078263a
    Collecting pandas
      Downloading pandas-1.3.4-cp38-cp38-manylinux_2_17_x86_64.manylinux2014_x86_64.whl (11.5 MB)
    Collecting numpy
      Downloading numpy-1.21.3-cp38-cp38-manylinux_2_12_x86_64.manylinux2010_x86_64.whl (15.7 MB)
    Collecting category_encoders
      Downloading category_encoders-2.3.0-py2.py3-none-any.whl (82 kB)
    Collecting sklearn
      Downloading sklearn-0.0.tar.gz (1.1 kB)
    Collecting scipy
      Downloading scipy-1.7.1-cp38-cp38-manylinux_2_5_x86_64.manylinux1_x86_64.whl (28.4 MB)
    Collecting joblib
      Downloading joblib-1.1.0-py2.py3-none-any.whl (306 kB)
    Collecting pytz>=2017.3
      Downloading pytz-2021.3-py2.py3-none-any.whl (503 kB)
    Collecting python-dateutil>=2.7.3
      Downloading python_dateutil-2.8.2-py2.py3-none-any.whl (247 kB)
    Collecting scikit-learn>=0.20.0
      Downloading scikit_learn-1.0-cp38-cp38-manylinux_2_12_x86_64.manylinux2010_x86_64.whl (25.8 MB)
    Collecting patsy>=0.5.1
      Downloading patsy-0.5.2-py2.py3-none-any.whl (233 kB)
    Collecting statsmodels>=0.9.0
      Downloading statsmodels-0.13.0-cp38-cp38-manylinux_2_17_x86_64.manylinux2014_x86_64.whl (9.9 MB)
    Collecting six
      Downloading six-1.16.0-py2.py3-none-any.whl (11 kB)
    Collecting threadpoolctl>=2.0.0
      Downloading threadpoolctl-3.0.0-py3-none-any.whl (14 kB)
    Building wheels for collected packages: sklearn
      Building wheel for sklearn (setup.py): started
      Building wheel for sklearn (setup.py): finished with status 'done'
      Created wheel for sklearn: filename=sklearn-0.0-py2.py3-none-any.whl size=1309 sha256=d7ab26295e1ac905b0e094d089f736cc81c81f2b1d7857e5c371414206f9a776
      Stored in directory: /root/.cache/pip/wheels/22/0b/40/fd3f795caaa1fb4c6cb738bc1f56100be1e57da95849bfc897
    Successfully built sklearn
    Installing collected packages: six, pytz, python-dateutil, numpy, threadpoolctl, scipy, patsy, pandas, joblib, statsmodels, scikit-learn, sklearn, category-encoders
    Successfully installed category-encoders-2.3.0 joblib-1.1.0 numpy-1.21.3 pandas-1.3.4 patsy-0.5.2 python-dateutil-2.8.2 pytz-2021.3 scikit-learn-1.0 scipy-1.7.1 six-1.16.0 sklearn-0.0 statsmodels-0.13.0 threadpoolctl-3.0.0
     ---> 7f62f9b2db86
    Step 12/13 : COPY default_modeling default_modeling
     ---> a74f7163a3d7
    Step 13/13 : ENTRYPOINT ["python3"]
     ---> Running in cf38eaafbce6
    Removing intermediate container cf38eaafbce6
     ---> 413468d329ce
    Successfully built 413468d329ce
    Successfully tagged default_model:latest



## First, run unit test in Image to make sure everything's OK


```python
First, run unit test in Image to make sure everything's OK
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
                                                        -m default_modeling.train --train-file train_set_1.csv
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
                                                        -m default_modeling.train --train-file train_set_2.csv
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
