import logging
import re

from typing import Union
import pathlib

import numpy as np
import pandas as pd

LOGGER = logging.getLogger(__name__)


def load_data(event_data: Union[list, str]) -> pd.DataFrame:
    
    """Takes the data returned from Cassadra queries and converts them into a
    DataFrame that can be digested.

    Args:
      event_data(list[dict] or string): The data returned from sedds Cassandra client fetch method or the name of a csv file
    Returns:
     pd.DataFrame
    """

    if not event_data:
        LOGGER.error("event_data is empty")
        return pd.DataFrame()

    if isinstance(event_data, str) or isinstance(event_data, pathlib.PosixPath):
        data = pd.read_csv(event_data)
    else:
        data = pd.DataFrame(event_data)

    return data
