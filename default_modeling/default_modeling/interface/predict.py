import argparse
import joblib
import os
import time
import pandas as pd
from scipy.stats import ks_2samp 
from sklearn import metrics
import numpy as np

from ..utils.preproc import CategoricalEncoder
from ..utils.preproc import NumericEncoder
from ..utils.preproc import feature_definition

import argparse
import joblib
import os
import time
import errno

import pandas as pd
from scipy.stats import ks_2samp 
from sklearn import metrics
import numpy as np

from ..utils.preproc import CategoricalEncoder
from ..utils.preproc import NumericEncoder
from ..utils.preproc import feature_definition

def predict():
    
    """
    Args:
    Returns:
    Raise: FileNotFoundError if model hasn't been found
    """
    
    print("extracting arguments")
    parser = argparse.ArgumentParser()
    # Data, model, and output directories
    parser.add_argument("--model-dir", type=str, default=os.environ.get("MODEL_DIR"))
    parser.add_argument("--test-folder", type=str, default=os.environ.get("TEST_FOLDER"))
    parser.add_argument("--model-name", type=str, default=os.environ.get("MODEL_NAME"))
    parser.add_argument("--target", type=str, default=os.environ.get("TARGET"))
    parser.add_argument("--test-file", type=str, default=os.environ.get("TEST_FILE"))

    args, _ = parser.parse_known_args()

    model_path = os.path.join(args.model_dir, args.model_name)
    model_file = f"{model_path}.joblib"
    print(args)
    if not os.path.isfile(model_file):
        raise(FileNotFoundError(errno.ENOENT, os.strerror(errno.ENOENT), model_file))
        
    print(f"Found model at: {model_file}")
    
    risk_model = joblib.load(model_file)
    test_df = pd.read_csv(os.path.join(args.test_folder, args.test_file))
    categories_features, numerics_features = feature_definition()
    all_features = categories_features + numerics_features + ["default"]
    print(f"Predicting {args.test_file} ....")
    start_time = time.time()
    y_test_pred = risk_model.predict_proba(test_df[all_features])
    print(f"Finish after {time.time() - start_time} s")
    y_test_pred = y_test_pred[:, 1]
    test_df["default_prediction"] = y_test_pred
    
    saved_filed = os.path.join(args.test_folder, f"{args.test_file}")
    print(f"...to csv {saved_filed}")
    test_df.to_csv(saved_filed, index=False)
    
if __name__ == "__main__":
    predict()
