# Test of tree classifiers
# Three levels of tree classifiers:
#%%

typedataProc = input("Type of DataProcessing:\n1. Dirty Data\n2. Preprocess Data\n3. DEFCON\n4. Feature Extraction\n")


vocabSi = input("Vocabsize: ")
vocabSi = int(vocabSi)

print('Importing libraries...')
import pandas as pd
import numpy as np
import os

from sklearn.model_selection import train_test_split

import classifier_funcs as cf
import get_data as gd
import wordstonumbers as wtn
import preprocessing_space as ps
import lowercasetext as lct
import DEFCON

import feature_extraction as fe

print('Done')

#%%

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

    ############################# Vectorizing Data #############################

    print("\nVectorizing & Splitting Data...")
    X, vec = ps.preprocess_text(df_in, ['title', 'text'], max_features=vocabSi, print_vocabulary=False)

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

elif typedataProc == '2':

    df = pd.read_csv('WELFake_Dataset.csv')
    # Replace NaNs with empty string in 'title' and 'text' columns
    df['title'] = df['title'].fillna('')
    df['text'] = df['text'].fillna('')
    X_title = df['title']
    X_text = df['text']
    y = df['label']

    X_textlower = lct.Lowercase(X_text)
    X_text = ps.preprocess_space(X_textlower)
    X_titlelower = lct.Lowercase(X_title)
    X_title = ps.preprocess_space(X_titlelower)
    X_tex = ps.remove_stopwords(X_text)
    X_tit = ps.remove_stopwords(X_title)

    #vocab_d = wtn.get_vocab(X_text, n_words=10000)
    df_in = pd.DataFrame({'title': X_tit, 'text': X_tex})

    ############################# Vectorizing Data #############################

    print("\nVectorizing & Splitting Data...")
    X, vec = ps.preprocess_text(df_in, ['title', 'text'], max_features=vocabSi, print_vocabulary=False)

    print(X.shape)

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


elif typedataProc == '3':

    X_title_final, X_text_final, y = DEFCON.DEFCON5(NN = False,save_to_file=True)

    print("\nSplitting data...")

    df_in = pd.DataFrame({'title': X_title_final, 'text': X_text_final})

    ############################# Vectorizing Data #############################

    print("\nVectorizing & Splitting Data...")
    X, vec = ps.preprocess_text(df_in, ['title', 'text'], max_features=vocabSi, print_vocabulary=False)

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

elif typedataProc == '4':
    print('lol, not done yet')

    df = pd.read_csv('WELFake_Dataset.csv')
    # Replace NaNs with empty string in 'title' and 'text' columns
    df['title'] = df['title'].fillna('')
    df['text'] = df['text'].fillna('')
    X_title = df['title']
    X_text = df['text']
    y = df['label']

    X_textlower = lct.Lowercase(X_text)
    X_text = ps.preprocess_space(X_textlower)
    X_titlelower = lct.Lowercase(X_title)
    X_title = ps.preprocess_space(X_titlelower)
    X_tex = ps.remove_stopwords(X_text)
    X_tit = ps.remove_stopwords(X_title)

    #vocab_d = wtn.get_vocab(X_text, n_words=10000)
    df_in = pd.DataFrame({'title': X_tit, 'text': X_tex})

    df_out, strings = fe.flag_vocab(df_in, y)

    print(df_out.shape)

    ############################# Vectorizing Data #############################

    print("\nVectorizing & Splitting Data...")
    X, vec = ps.preprocess_text(df_out, ['title', 'text'], strings, max_features=vocabSi, print_vocabulary=False)

    print(X.shape)

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


############################# Tree classifiers #############################


print("\nTraining Decision Tree Classifier...")
# Decision Tree Classifier
accuracy, bce_loss, conf_matrix, roc_auc, y_pred, y_pred_prob, clf = cf.LightGBM(X_train, y_train, X_test, y_test)

plots = input('Plot the Evaluation? (y/n): ')

if plots == 'y' or plots == '':
    cf.plot_confusion_matrix_and_roc(y_test, y_pred, y_pred_prob, typedataProc, vocabSi, data = [accuracy, bce_loss, roc_auc, vocabSi])

plot_learn = input('Plot Learning Curve? (y/n): ')

if plot_learn == 'y' or plot_learn == '':
    cf.plot_learning_curve(clf, X, y, cv = 5, num=20)

    #cf.cross_val(clf, X, y)