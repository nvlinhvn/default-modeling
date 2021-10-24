
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
