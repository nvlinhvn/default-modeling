
from typing import Union

import pandas as pd
import numpy as np
from category_encoders.woe import WOEEncoder

def feature_definition():
    
    numerics = ['age', 'account_amount_added_12_24m', 'account_days_in_rem_12_24m', 'account_days_in_term_12_24m',
                'account_incoming_debt_vs_paid_0_24m', 'avg_payment_span_0_12m', 'avg_payment_span_0_3m',
                'max_paid_inv_0_12m', 'max_paid_inv_0_24m', 'num_active_div_by_paid_inv_0_12m',
                'num_active_inv', 'num_arch_dc_0_12m', 'num_arch_dc_12_24m', 'num_arch_ok_0_12m', 
                'num_arch_ok_12_24m', 'num_arch_rem_0_12m', 'num_arch_written_off_12_24m',
                'num_unpaid_bills', 'sum_capital_paid_account_0_12m', 'sum_capital_paid_account_12_24m',
                'has_paid',
                'sum_paid_inv_0_12m', 'time_hours']
    categories = ['account_status', 'account_worst_status_0_3m', 'account_worst_status_12_24m',
                  'account_worst_status_3_6m', 'account_worst_status_6_12m', 'status_last_archived_0_24m',
                  'status_2nd_last_archived_0_24m', 'status_3rd_last_archived_0_24m', 'status_max_archived_0_6_months',
                  'status_max_archived_0_12_months', 'status_max_archived_0_24_months',
                  'worst_status_active_inv', 'merchant_category', 'merchant_group', 'name_in_email']
    
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
            bucket_list = range(0, 100 + bin_width, bin_width)
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
        if "has_paid" in X:
            # has_paid is boolean
            X["has_paid"] = X["has_paid"].astype(int)

        for col in self.column_list:        
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
