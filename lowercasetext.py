import copy
import pandas as pd


def lowercase(string):
    return str(string).lower()

def Lowercase(dataframe):
    return dataframe.map(lowercase)