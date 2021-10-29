"""Base class for unit tests"""

import copy
import logging
import unittest

import joblib
import pandas as pd
import pathlib

from default_modeling.utils.load import load_data

LOGGER = logging.getLogger(__name__)


class TestWithData(unittest.TestCase):
    raw = dict()
    available_file = None

    @classmethod
    def setUpClass(cls) -> None:
        cls.available_file = cls.get_available_file()

    @classmethod
    def get_available_file(cls) -> list:
        """Returns the list available test data
        e.g. everything stored under tests/data
        Args:
        Returns: List file
        """
        p = pathlib.Path(".")
        test_data = pathlib.Path("default_modeling/tests/data/").glob("*.csv")
        test_data = [f for f in test_data]
        print("Found the following test data")
        for f in test_data:
            print(f)

        return test_data

    @classmethod
    def get_raw(cls, file: str) -> pd.DataFrame:
        """Lazy loading for raw data; will return a copy of the df

        Args:
          file: file name of sample test

        Returns:

        """

        if file in cls.raw:
            LOGGER.info("Found raw data for %s", file)
            return cls.raw[file].copy()
        
        df = load_data(file)

        LOGGER.info("Adding raw data for %s", file)
        cls.raw[file] = df

        return df.copy()
