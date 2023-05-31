import copy
import pandas as pd


def lowercase(string):
    return str(string).lower()

def execute(dataframe):
    return dataframe.map(lowercase)