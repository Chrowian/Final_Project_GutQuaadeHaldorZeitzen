# Testing data preprocessing on the neural network
#%%
print("Importing libraries...")

import numpy as np
import pandas as pd

#from wordstonumbers import sentence_to_integer_sequence
import get_data as gd
import wordstonumbers as wtn
import preprocessing_space as ps
import lowercasetext as lct
from oldshiz.rnn_lstm_func import RNN_LSTM
import classifier_funcs as cf
import tokeniser as tok


#%%
print("Importing data...")

# Importing the dataset
X_title, _ = gd.get_title_data()

X_text, y = gd.get_text_data()

#%%
input = input("Do you want to preprocess the data? (y/n): ")
if input == 'y' or input == '':
    print("Preprocessing data...")

    X_textlower = lct.Lowercase(X_text)
    X_text = ps.preprocess_space(X_textlower)
    X_titlelower = lct.Lowercase(X_title)
    X_title = ps.preprocess_space(X_titlelower)
    vocab_d = wtn.get_vocab(X_text)

    # # Combining title and text to dataframe
    df = pd.DataFrame({'title': X_title, 'text': X_text, 'label': y})

if input == 'n':
    df = pd.DataFrame({'title': X_title, 'text': X_text, 'label': y})

# Tokenising the data
print("Tokenising data...")

df2 = tok.tokenize_dataframe(df, vocab_size=50000, max_length=300)
data = df2['text']
labels = df2['label'].values
#%%
data_np = np.array(data.tolist())

#%%
type_calssif = input("Which classifier do you want to use? (1: RNN-LSTM or 2: LightGBM GBDT): ")

if type_calssif == '1':
    print('Running RNN-LSTM...')

    # Running RNN-LSTM
    cf.RNN_LSTM(data_np, labels, epoch=20, max_features=50000, batchsize= 256*8)

if type_calssif == '2':
    print('Running LightGBM GBDT...')

    # Running LightGBM GBDT
    cf.LightGBM_GBDT(data_np, labels, epoch=20, max_features=50000, batchsize= 256*8)
# %
# %%
