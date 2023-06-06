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
from rnn_lstm_func import RNN_LSTM

#%%
print("Importing data...")

# Importing the dataset
X_title, _ = gd.get_title_data()

X_text, y = gd.get_text_data()

#%%
print("Preprocessing data...")

X_textlower = lct.Lowercase(X_text)
X_text = ps.preprocess_space(X_textlower)
X_titlelower = lct.Lowercase(X_title)
X_title = ps.preprocess_space(X_titlelower)
vocab_d = wtn.get_vocab(X_text)

# Combining title and text to dataframe
df = pd.DataFrame({'title': X_title, 'text': X_text, 'label': y})
df['total'] = df['title'] + ' ' + df['text']

labels = df['label'].values

#%%
print('Running RNN-LSTM...')

# Running RNN-LSTM
RNN_LSTM(df, labels, 20, 2048, 2)
# %%
