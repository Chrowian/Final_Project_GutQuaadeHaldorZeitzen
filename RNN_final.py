# Test of RNN LSTM classifiers

typedataProc = input("Type of DataProcessing:\n1. Dirty Data\n2. Preprocess Data\n3. DEFCON\n")


vocabSi = input("Vocabsize: ")
vocabSi = int(vocabSi)

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



if typedataProc == '1':
    print("\nImporting data...")
    
    df = pd.read_csv('WELFake_Dataset.csv')
    # Replace NaNs with empty string in 'title' and 'text' columns
    df['title'] = df['title'].fillna('')
    df['text'] = df['text'].fillna('')
    X_title = df['title']
    X_text = df['text']
    y = df['label']

    

    print("\nSplitting data...")

    df_in = pd.DataFrame({'title': X_title, 'text': X_text})


    ############################# Tokenising Data #############################

    print("\nTokenising Data...")
    df2 = tok.tokenize_dataframe(df_in, vocab_size=vocabSi, max_length=300)

    labels = y

    df2['total'] = df2['title'] + df2['text']

    #data = df2['total']
    data = df2['text']

    data_np = np.array(data.tolist())   # Convert to numpy array

elif typedataProc == '2':

    X_title, _ = gd.get_title_data()

    X_text, y = gd.get_text_data()

    print("Preprocessing data...")

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
    df2 = tok.tokenize_dataframe(df_in, vocab_size=vocabSi, max_length=300)

    labels = y

    df2['total'] = df2['title'] + df2['text']

    #data = df2['total']
    data = df2['text']

    data_np = np.array(data.tolist())   # Convert to numpy array


elif typedataProc == '3':

    X_title_final, X_text_final, y = DEFCON.DEFCON5(NN = False,save_to_file=True)

    print("\nSplitting data...")

    df_in = pd.DataFrame({'title': X_title_final, 'text': X_text_final})

    ############################# Tokenising Data #############################

    print("\nTokenising Data...")
    df2 = tok.tokenize_dataframe(df_in, vocab_size=vocabSi, max_length=300)

    labels = y

    df2['total'] = df2['title'] + df2['text']

    #data = df2['total']
    data = df2['text']

    data_np = np.array(data.tolist())   # Convert to numpy array

############################# RNN classifiers #############################


print("\nTraining RNN-LSTM...")
# Decision Tree Classifier
cf.RNN_LSTM(data_np, labels, epoch = 20, batchsize = 258 * 8, max_features=vocabSi, names = typedataProc, plotIT=True, train = 'y')
