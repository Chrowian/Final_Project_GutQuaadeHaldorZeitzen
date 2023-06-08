# script for testing the lgbm model
#%%
import pandas as pd
import numpy as np
from scipy.sparse import hstack
from sklearn.feature_extraction.text import TfidfVectorizer

from sklearn.model_selection import train_test_split
import classifier_funcs as cf
import get_data as gd
import wordstonumbers as wtn
import preprocessing_space as ps
import lowercasetext as lct



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
vocab_d = wtn.get_vocab(X_text, n_words=10000)

df_in = pd.DataFrame({'title': X_title, 'text': X_text})
#%%
X, vec = ps.preprocess_text(df_in, ['title', 'text'])
#%%

# make input data, drop labels from dataframe
y = pd.DataFrame({'label': y})
y = y['label']


# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Now you can pass these to your LightGBM classifier function
# %%

cf.LightGBM(X_train, y_train, X_test, y_test)
# %%
