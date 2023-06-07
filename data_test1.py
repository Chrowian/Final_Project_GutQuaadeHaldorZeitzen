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
    vocab_d = wtn.get_vocab(X_text, n_words=10000)

    # # Combining title and text to dataframe
    df = pd.DataFrame({'title': X_title, 'text': X_text, 'label': y})
    # make combined text
    #df['total'] = df['title'] + ' ' + df['text']

if input == 'n':
    df = pd.DataFrame({'title': X_title, 'text': X_text, 'label': y})
    #df['total'] = df['title'] + ' ' + df['text']

# Tokenising the data
print("Tokenising data...")

df2 = tok.tokenize_dataframe(df, vocab_size=10000, max_length=300)
#df2 = tok.tokenize_vocab(df, vocab_d, max_length=300)
data = df2['text']
labels = df2['label'].values
#%%
data_np = np.array(data.tolist())

#%%

print('Running RNN-LSTM...')

# Running RNN-LSTM
cf.RNN_LSTM(data_np, labels, epoch=20, max_features=10000, batchsize= 256*8)
# %
# %%
