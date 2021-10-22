#default_modeling/default_modeling/interface/train.py

import argparse
import joblib
import os
import pathlib

import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.utils import class_weight

from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.compose import ColumnTransformer

from .default_modeling.utils.preproc import CategoricalEncoder
from .default_modeling.utils.preproc import NumericEncoder
from .default_modeling.utils.preproc import feature_definition

import warnings
warnings.filterwarnings("ignore")


def train():
        
    print("extracting arguments")
    parser = argparse.ArgumentParser()


    # Data, model, and output directories
    parser.add_argument("--model-dir", type=str, default=os.environ.get("MODEL_DIR"))
    parser.add_argument("--datafolder", type=str, default="./train_data/")
    parser.add_argument("--model-name", type=str, default=os.environ.get("MODEL_NAME"))
    parser.add_argument("--train-file", type=str, default="train_set.csv")
    parser.add_argument("--n-estimators", type=int, default=100)
    parser.add_argument("--min-samples-leaf", type=int, default=10)
    parser.add_argument("--max-depth", type=int, default=10)  
    parser.add_argument("--random-state", type=int, default=1234)   
    parser.add_argument(
        "--target", type=str, default='default'
    )


    args, _ = parser.parse_known_args()

    print("Training Data Preparation")
    print(os.path.join(args.datafolder, args.train_file))
    train_df = pd.read_csv(os.path.join(args.datafolder, args.train_file))
    y_train = train_df[args.target]
     
    print(args)
    categories_features, numerics_features = feature_definition()
    all_features = categories_features + numerics_features + [args.target]
    categories_features.append(args.target)
    input_features = categories_features + numerics_features
    print("Total Input Features", len(input_features))

    # Preproc Data
    numeric_encoder = NumericEncoder
    numeric_transformer = Pipeline(steps=[
        ('numeric_encoder', numeric_encoder(column_list=numerics_features, 
                                            bin_width=1))])
    
    categorical_encoder = CategoricalEncoder
    categorical_transformer = Pipeline(steps=[
    ('categorical_encoder', categorical_encoder(column_list=categories_features))])

    preprocessor = ColumnTransformer(
        transformers=[
            ('cat', categorical_transformer, categories_features),
            ('num', numeric_transformer, numerics_features)],
        remainder="drop")
    
    class_weight_list = class_weight.compute_class_weight(class_weight='balanced',
                                                          classes=np.unique(y_train),
                                                          y=y_train)
    class_weight_dict = {}
    for i, weight in enumerate(class_weight_list):
        class_weight_dict[i] = weight
    print('class weight', class_weight_dict)

    rf_model = RandomForestClassifier(
                n_estimators=args.n_estimators, 
                min_samples_leaf=args.min_samples_leaf, 
                max_depth=args.max_depth, 
                class_weight=class_weight_dict,
                random_state=args.random_state,
                n_jobs=-1,
    )
    
    ml_pipeline = Pipeline(steps=[
        ("preprocessor", preprocessor),
        ("model", rf_model)
    ])
    
    ml_pipeline.fit(train_df[all_features], y_train)
    print("Saving Pipeline ........")
    os.makedirs(args.model_dir, exist_ok=True)
    joblib.dump(ml_pipeline, os.path.join(args.model_dir, args.model_name))
    print("Congratulation! Finish.")    
         

if __name__ == "__main__":
    train()
