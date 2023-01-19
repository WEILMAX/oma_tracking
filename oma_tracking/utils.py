"""
utils.py is a module for helper functions in the oma_tracking package.
-------------------------------------------------------------------------
Author: Maximillian Weil
"""

import pandas as pd


def check_columns(cols: list, data: pd.DataFrame) -> bool:
    """
    Check if all elements of a list are in the columns of a dataframe.

    Args:
        columns (list): The list of columns to be checked
        data (pd.DataFrame): The dataframe to be checked against

    Returns:
        bool: True if all elements of the list are in the columns of the dataframe, False otherwise
    """
    return all(col in data.columns for col in cols)
