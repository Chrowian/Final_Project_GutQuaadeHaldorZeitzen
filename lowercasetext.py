import copy
import pandas as pd
"""def lowercasetext(data,n_to_change='all',filepath='',filename='data_lowercase.csv'):
    
    Changes uppercase letters to lowercase in text only and saves 
    arg data must be a pandas dataframe with the same columns as in our original datafile
    kwarg n_to_change = {'all' or int} is how many indices for which we change stuff
    
    N=n_to_change
    if n_to_change=='all':
        N=len(d2)
    d2 = copy.deepcopy(data)
    for i in range(N):
        d2.iloc[i]['text'] = data.iloc[i]['text']
    d2.rename(columns={'Unnamed: 0': ''})
    d2.to_csv(filepath+filename,index=False)"""

def lowercase(string):
    return string.lower()

def execute(dataframe):
    return dataframe.map(lowercase)