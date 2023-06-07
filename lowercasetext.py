import pandas as pd

def lowercase(string):
    return str(string).lower()

def Lowercase(dataframe):
    try:
        return dataframe.map(lowercase)
    except:
        #return dataframe.apply(np.vectorize(lowercase),**kwargs)
        return dataframe.applymap(lowercase)