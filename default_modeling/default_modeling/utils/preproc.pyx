from typing import Union

import pandas as pd
import numpy as np
from category_encoders.woe import WOEEncoder

def feature_definition():
    
    """
    Define list of categorical/numerical features
    """
    # categories
    categories = ['category_1', 'category_2', 'category_3', 'category_4', 'category_5', 'category_6', 'category_7',
                  'category_8', 'category_9', 'category_10', 'category_11', 'category_12', 'category_13', 'category_14',
                  'category_15']
    # numerics
    numerics = ['numeric_0', 'numeric_1', 'numeric_2', 'numeric_3', 'numeric_4', 'numeric_5', 'numeric_6', 'numeric_7',
                'numeric_8', 'numeric_9', 'numeric_10', 'numeric_11', 'numeric_12', 'numeric_13', 'numeric_14', 
                'numeric_15', 'numeric_16', 'numeric_17', 'numeric_18', 'numeric_19', 'numeric_20', 'numeric_21',
                'numeric_22']
    
    return categories, numerics


# ----- PREPROC ------
class NumericEncoder():
    
    """
    Encode number by binning into different ranges
    """
    
    def __init__(self, 
                 column_list: list = None,
                 bin_width: int = None):
        
        self.column_list = column_list
        self.bin_width = bin_width
        
    def __binning__(self, 
                    X: pd.Series, 
                    bucket_list: list, 
                    bin_width: int) -> list:
        """
        Helper function to bin a series
        Args:
            X: continuous value Series
            bucket_list: list of different value for each bin
                         Some features require specific binning values
            bin_width: auto-bin with width percentage
            (Either bin_width or bucket_list is used)
        Returns:
            list of binned values
        """
        
        X = X.copy(deep=True)
        n_null = X.isna().sum()

        if n_null > 0:
            X = X.fillna(-1)
            bucket_bin = [-1]
        else:
            bucket_bin = []

        if bucket_list is None:
            bucket_list = list(range(0, 100 + bin_width, bin_width))
            for i, q in enumerate(bucket_list):
                q_quantile = round(np.percentile(X.astype(np.float32).values, q), 3)
                if q_quantile not in bucket_bin:
                    bucket_bin.append(q_quantile)

        else:
            bucket_bin = bucket_bin + list(bucket_list)
            
        return bucket_bin
    
    def fit(self, 
            X: pd.DataFrame, 
            y: Union[list, np.array] = None, 
            verbose: int = 0):
        """
        Construct encoder as a dictionary
        Args:
            X: pd.DataFrame
            y: np.array Output
            verbose: int. for logging info
        Return:
            encoder object
        """
        X = X.copy(deep=True)
        encode_dict = {}
        for column in self.column_list:
            if column != "age":
                # Encode other columns
                encode_dict[column] = self.__binning__(X[column], None, self.bin_width)
            elif column == "age":
                # Specific encoding for age columns
                max_age = max(X[column])
                age_bucket=[0, 18, 24, 40, 57, 75, max_age]
                encode_dict[column] = self.__binning__(X[column], age_bucket, self.bin_width)
            if verbose:
                print('\n', column)
                print(encode_dict)
        self.encoder = encode_dict
        return self
        
    def transform(self, 
                  X: pd.DataFrame) -> pd.DataFrame:
        """
        Use built encode to transform data
        Args:
            X: pd.DataFrame
        Return:
            pd.DataFrame with transformed columns
        """
        X = X.copy(deep=True)

        for col in self.column_list:     
            # if boolean, convert to int
            if X[col].dtype == bool:
                X[col] = X[col].astype(int)
            
            if X[col].isnull().any():
                X[col] = X[col].fillna(-1)
            
            bucket_bin = self.encoder[col]

            # Extend bin range if values exceed
            if max(bucket_bin) < max(X[col]):
                bucket_bin[-1] = max(X[col])
            if  min(bucket_bin) > min(X[col]):
                bucket_bin[0] = min(X[col])

            X[col] = pd.cut(X[col],
                            bucket_bin,
                            include_lowest=True,
                            retbins=True,
                            labels=bucket_bin[:-1])[0].astype(float)
        return X

class CategoricalEncoder():

    """
    Encode categories by Weight of Evidence 
    (from category_encoders library)
    """
    
    def __init__(self, 
                 column_list: list = None):
        self.encoder = None
        self.column_list = column_list
    
    def fit(self, 
            X: pd.DataFrame,
            y: Union[list, np.array],
            verbose: int = 0):
        """
        Construct encoder as a dictionary
        Args:
            X: pd.DataFrame
            y: np.array Output
            verbose: int. for logging info
        Return:
            encoder object
        """
        X = X.copy(deep=True)
        woe_encoder = WOEEncoder(cols=self.column_list, random_state=50)
        woe_encoder = woe_encoder.fit(X[self.column_list], y)
        self.encoder = woe_encoder
        return self
                        
    def transform(self, 
                  X: pd.DataFrame) -> pd.DataFrame:
        """
        Use built encode to transform data
        Args:
            X: pd.DataFrame
        Return:
            pd.DataFrame with transformed columns
        """
        X = X.copy(deep=True)
        X[self.column_list] = self.encoder.transform(X[self.column_list])
        return X
