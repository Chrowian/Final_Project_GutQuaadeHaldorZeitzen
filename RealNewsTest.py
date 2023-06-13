#%%
print("Importing libraries...")

import numpy as np
import pandas as pd

#from wordstonumbers import sentence_to_integer_sequence
import get_data as gd
import preprocessing_space as ps
import lowercasetext as lct
import classifier_funcs as cf
import tokeniser as tok
from sklearn.model_selection import train_test_split

import pandas as pd
import numpy as np
from tensorflow import keras
from keras.models import Sequential
from keras.layers import Dense, Embedding, LSTM
from sklearn.model_selection import train_test_split

from keras.models import load_model
from keras.callbacks import ModelCheckpoint
from keras import regularizers
import matplotlib.pyplot as plt

#%%

X_title, _ = gd.get_title_data()

X_text, y = gd.get_text_data()

print("Preprocessing data...")

#%%

# Link to real article: https://www.bbc.com/news/world-us-canada-65875898

# text from the article
text = """Mr Barr criticised his former boss and said he had no right to keep the files allegedly found at his home. Mr Trump will appear in court in Miami on Tuesday to face dozens of charges accusing him of illegally retaining classified information. He has repeatedly denied wrongdoing. Speaking to Fox News on Sunday, Mr Barr, who was Mr Trump's attorney general from February 2019 until December 2020, defended the 37-count indictment made public by Special Counsel Jack Smith on Friday. "I was shocked by the degree of sensitivity of these documents and how many there were... and I think the counts under the Espionage Act that he wilfully retained those documents are solid counts," he said. "If even half of it is true, then he's toast. It's a very detailed indictment and it's very damning," he added. The 73-year-old was once one of Mr Trump's staunchest allies, but has been increasingly critical of him since leaving office. Shortly after he gave the interview, Mr Trump described him as a "'disgruntled former employee" and "lazy attorney general who was weak [and] totally ineffective". Many prominent Republicans have been hesitant to criticise the former president, who is the frontrunner to be the party's presidential candidate in 2024, and have instead targeted the justice department and the broader investigation."""

text2 = """NEW YORK—Paying out of pocket to print his reaction to a highly publicized court case in four area newspapers, former President Donald Trump took out full-page ads Tuesday calling for New York to reinstate capital punishment and sentence him to death following his historic indictment. “This person must serve as an example so that others will think long and hard before committing a crime,” read the ad in which Trump, without directly naming himself as the defendant who should be executed by lethal injection, urged readers of The New York Times, the Daily News, the New York Post, and Newsday to show him no mercy. “I am not looking to psychoanalyze or understand this person. I am looking to punish him. Let our politicians give back law enforcement’s power to keep us safe. We must cease our continuous pandering to the criminal population of this city. Send a message loud and clear to those who would terrorize New York—BRING BACK THE DEATH PENALTY!” Trump went on to call himself a “depraved animal who needs to be put down immediately.”"""
# Create a dictionary
data_real = {'text': [text, text2]}

# Create DataFrame
df_real = pd.DataFrame(data_real)

print(df_real)

real_text = df_real['text']

real_lower = lct.Lowercase(real_text)
real_text = ps.preprocess_space(real_lower)

#%%
X_textlower = lct.Lowercase(X_text)
X_text = ps.preprocess_space(X_textlower)
X_titlelower = lct.Lowercase(X_title)
X_title = ps.preprocess_space(X_titlelower)
#vocab_d = wtn.get_vocab(X_text, n_words=10000)

#df['total'] = df['title'] + ' ' + df['text']
############################# Vectorizing Data #############################

df_in = pd.DataFrame({'title': X_title, 'text': X_text})

df_real_in = pd.DataFrame({'text': real_text})

############################# Tokenising Data #############################
#%%
print("\nTokenising Data...")
df2, real_tokenised = tok.tokenize_dataframe(df_in, real_news=df_real_in, vocab_size=10000, max_length=300)
#%%
labels = y

df2['total'] = df2['title'] + df2['text']

#data = df2['total']
data = df2['text']

data_np = np.array(data.tolist())   # Convert to numpy array
# %%
X_train, X_val, y_train, y_val = train_test_split(data_np, labels, test_size=0.3, random_state=42)

X_val, X_test, y_val, y_test = train_test_split(X_val, y_val, test_size=0.5, random_state=42)
#%%
model = Sequential()
model.add(Embedding(10000, 32)) 
model.add(LSTM(32, dropout=0.2, recurrent_dropout=0.2)) 
model.add(Dense(1, activation='sigmoid', kernel_regularizer=regularizers.l2(0.01))) 

batch_size = 258*8
epochs = 15

optimize = keras.optimizers.legacy.Adam(lr = 0.001)

model.compile(loss='binary_crossentropy',
            optimizer=optimize,
            metrics=['accuracy'])

print(model.summary())

# Specify the path where you want to save the model
filepath = f"Result_Models/RNN_MODELS/test_real.h5"

# Initialize ModelCheckpoint
checkpoint = ModelCheckpoint(filepath, monitor='val_loss', verbose=1, save_best_only=True, mode='min')

# Pass the ModelCheckpoint callback to model.fit() function
history = model.fit(X_train, y_train, 
                    batch_size=batch_size, 
                    epochs=epochs, 
                    validation_data=(X_val, y_val),
                    callbacks=[checkpoint])

fig, ax = plt.subplots(figsize=(7,4))
ax.plot(history.history['loss'], label='Train', color = 'k')
ax.plot(history.history['val_loss'], label='Validation', color = 'b')
ax.set_title('Model Loss vs Epoch')
ax.set_ylabel('Loss')
ax.set_xlabel('Epoch')
ax.set_xticks(np.arange(0, 15, 5))

plt.tight_layout()

plt.show()
#%%
model_best = load_model(f'Result_Models/RNN_MODELS/test_real.h5')

loss, accuracy, y_pre, y_pred, roc_auc = cf.evaluate_model(model_best, X_test, y_test, plot = False)

# %%
# Extract 'text' column from the dataframe and convert it to a numpy array
real_tokenised_array = np.stack(real_tokenised['text'])

# Use the numpy array for prediction
y_pred_real = model_best.predict(real_tokenised_array)
# %%
print('Classification of the Articles: 0: Real, 1: Fake\n')
print(f'Article 1 (BBC):       {np.round(y_pred_real[0])} ({y_pred_real[0]})')
print(f'Article 2 (The Onion): {np.round(y_pred_real[1])} ({y_pred_real[1]})')
# %%
