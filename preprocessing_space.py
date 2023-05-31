import pandas as pd

# Define the special characters you want to add spaces before and after
special_chars = r'([!@#$%^&*()_\-+=<>?.,:;\'"/\\])'

def preprocess_space(dataframe):
    """
    :param - dataframe: pandas dataframe X (text)
    :return: pandas dataframe X (text) WITH SPACES around special characters
    """
    dataframe = dataframe.str.replace(special_chars, r' \1 ')
    return dataframe