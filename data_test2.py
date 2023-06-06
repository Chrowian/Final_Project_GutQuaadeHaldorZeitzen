# Testing data preprocessing on Transformer
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

from transformers import BertTokenizer, TFBertForSequenceClassification
from transformers import BertConfig
from sklearn.model_selection import train_test_split
import tensorflow as tf
import pandas as pd

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

#%%

print("Tokenising data...")

# split your dataset into training and validation sets
train_df, val_df = train_test_split(df, test_size=.2)

# tokenize the dataframes
train_dataset = tok.tokenize_transformer(train_df)
val_dataset = tok.tokenize_transformer(val_df)


#%%

print('Running Transformer...')

config = BertConfig.from_pretrained('bert-base-uncased', num_labels=2)
config.return_dict = False
model = TFBertForSequenceClassification.from_pretrained('bert-base-uncased', config=config)

# %%
# define your batch size
batch_size = 256 * 8

# create your batched dataset
train_dataset = train_dataset.shuffle(1000).batch(batch_size)


optimizer = tf.keras.optimizers.Adam(learning_rate=5e-5)


def compute_loss(labels, predictions):
    logits = predictions[1]
    return tf.keras.losses.sparse_categorical_crossentropy(labels, logits, from_logits=True)

model.compile(optimizer=optimizer, loss=compute_loss, metrics=['accuracy'])

#%%
# fit your model
model.fit(train_dataset, epochs=3)

#%%
for example in train_dataset.take(1):  # Only take a single example
  print(example)

#%%

#%%
model.evaluate(val_dataset.batch(16))