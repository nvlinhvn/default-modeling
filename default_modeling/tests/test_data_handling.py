
import pandas.api.types as ptypes
from pandas.testing import assert_frame_equal


from tests.test_case_base import TestWithData
from default_modeling.utils.preproc import feature_definition
from default_modeling.utils.preproc import NumericEncoder
from default_modeling.utils.preproc import CategoricalEncoder


class DataHandlingTests(TestWithData):

    def test_load_data(self):
        """Validate that the most important columns are returned by loading function"""
        numerics, categories = feature_definition()
        key_column = numerics + categories + ["default"]
        for file in self.available_file:
            df = self.get_raw(file)
            for column in key_column:
                self.assertIn(column, df)

    def test_preproc_function(self):
        """Validate if the encoding return reasonable values:
        Categorical encoding: Only encode categorical columns
        Numeric encoding: Only encode numerical columns
        Expected result: all columns are numeric        
        """
        categories, numerics  = feature_definition()
        numeric_encoder = NumericEncoder(column_list=numerics, 
                                         bin_width=1)
        categorical_encoder = CategoricalEncoder(column_list=categories)
        
        input_column = numerics + categories

        for file in self.available_file:
            df = self.get_raw(file)
            y = df["default"].values
            categorical_encoder.fit(df, y)
            cat_transform = categorical_encoder.transform(df)
            # all transformed categories must be numerics
            assert all(ptypes.is_numeric_dtype(cat_transform[col]) for col in categories)
            # all numerical columns must be the same
            assert_frame_equal(cat_transform[numerics], df[numerics])
            
            numeric_encoder.fit(df.copy(), y)
            num_transfom = numeric_encoder.transform(df)
            # all transformed numerics must be numerics
            assert all(ptypes.is_numeric_dtype(num_transfom[col]) for col in numerics)
            # all categorical columns must be the same
            assert_frame_equal(num_transfom[categories], df[categories])
