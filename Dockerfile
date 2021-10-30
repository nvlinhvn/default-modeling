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
COPY default_modeling default_modeling

RUN pip install -r requirements.txt
RUN python3 -m default_modeling.setup build

ENTRYPOINT ["python3"]
