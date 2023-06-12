#%%
print("Importing libraries...")

import numpy as np
import pandas as pd

#from wordstonumbers import sentence_to_integer_sequence
import get_data as gd
import wordstonumbers as wtn
import preprocessing_space as ps
import lowercasetext as lct
import classifier_funcs as cf
import tokeniser as tok
from sklearn.model_selection import train_test_split
import DEFCON

#%%

X_title, _ = gd.get_title_data()

X_text, y = gd.get_text_data()

print("Preprocessing data...")

#%%

realnews = pd.read_csv('real_text.csv', error_bad_lines=False)

#%%
X_textlower = lct.Lowercase(X_text)
X_text = ps.preprocess_space(X_textlower)
X_titlelower = lct.Lowercase(X_title)
X_title = ps.preprocess_space(X_titlelower)
#vocab_d = wtn.get_vocab(X_text, n_words=10000)

#df['total'] = df['title'] + ' ' + df['text']
############################# Vectorizing Data #############################

df_in = pd.DataFrame({'title': X_title, 'text': X_text})

############################# Tokenising Data #############################

print("\nTokenising Data...")
df2 = tok.tokenize_dataframe(df_in, vocab_size=10000, max_length=300)

labels = y

df2['total'] = df2['title'] + df2['text']

#data = df2['total']
data = df2['text']

data_np = np.array(data.tolist())   # Convert to numpy array