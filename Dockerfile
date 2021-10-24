
FROM python:3.8
RUN pwd
RUN dir
ADD requirements.txt .
RUN pip install -r requirements.txt
ENV TRAINING_FOLDER=./train_data
ENV TESTING_FOLDER=./test_data
ENV MODEL_DIR=./default_modeling/default_modeling/interface/
ENV MODEL_NAME=risk_model.joblib

ADD default_modeling default_modeling
ADD train_data train_data

RUN dir
RUN python3 -m default_modeling.train --datafolder ${TRAINING_FOLDER} \
                                      --model-dir ${MODEL_DIR} \
                                      --model-name ${MODEL_NAME}
ENTRYPOINT ["python3", "-m"]
