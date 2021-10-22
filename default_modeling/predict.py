import argparse
import joblib
import os
import time
import pandas as pd
from scipy.stats import ks_2samp 
from sklearn import metrics
import numpy as np

from .default_modeling.utils.preproc import CategoricalEncoder
from .default_modeling.utils.preproc import NumericEncoder
from .default_modeling.utils.preproc import feature_definition

def predict():
    
    print("extracting arguments")
    parser = argparse.ArgumentParser()
    # Data, model, and output directories
    parser.add_argument("--model-dir", type=str, default=os.environ.get("MODEL_DIR"))
    parser.add_argument("--datafolder", type=str, default=os.environ.get("TESTING_FOLDER"))
    parser.add_argument("--model-name", type=str, default=os.environ.get("MODEL_NAME"))
    parser.add_argument("--test-file", type=str, default="test_set.csv")
 
    parser.add_argument(
        "--target", type=str, default="default"
    )

    args, _ = parser.parse_known_args()

    print(f"Model path: {os.path.join(args.model_dir, args.model_name)}")
    risk_model = joblib.load(os.path.join(args.model_dir, args.model_name))
    test_df = pd.read_csv(os.path.join(args.datafolder, args.test_file))
    categories_features, numerics_features = feature_definition()
    all_features = categories_features + numerics_features + ["default"]
    print(f"Predicting {args.test_file} ....")
    start_time = time.time()
    y_test_pred = risk_model.predict_proba(test_df[all_features])
    print(f"Finish after {time.time() - start_time} s")
    y_test_pred = y_test_pred[:, 1]
    test_df["default_prediction"] = y_test_pred
    saved_filed = os.path.join(args.datafolder, f"{args.test_file}")
    print(f"...to csv {saved_filed}")
    test_df.to_csv(saved_filed, index=False)
    
if __name__ == "__main__":
    predict()
