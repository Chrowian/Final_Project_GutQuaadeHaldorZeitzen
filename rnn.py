import pandas as pd
import numpy as np
from tensorflow import keras
from keras.preprocessing.text import Tokenizer
from keras.utils import pad_sequences
from keras.models import Sequential
from keras.layers import Dense, Embedding, LSTM, SpatialDropout1D, BatchNormalization
from sklearn.model_selection import train_test_split
from keras.utils.np_utils import to_categorical
from keras.callbacks import EarlyStopping
from keras.layers import Dropout
from keras.callbacks import LearningRateScheduler
import math
from keras.models import load_model
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc
from keras.callbacks import ModelCheckpoint
import seaborn as sns
from keras import regularizers

import re
import os

import time
import sys
import pickle
import matplotlib.pyplot as plt

font = {'family' : 'monospace'}
plt.rc('font', **font)

############################################# Load Data #############################################

# Load the data
print('\nReading in the data...')

df = pd.read_csv('WELFake_Dataset.csv')

print('\nData loaded.')

############################################# Preprocess Data ############################################# (Only a tiny biy i promise)

print('\nPreprocessing the data...')
# Preprocess the data: combine title and text, handle missing values, and clean the text
df['total'] = df['title'] + ' ' + df['text']
df['total'] = df['total'].apply(lambda x: x if type(x)==str else '')
df['total'] = df['total'].apply(lambda x: re.sub('[^a-zA-z0-9\s]', '', x))

print('\nData preprocessed.')

############################################# Tokenize Data #############################################

tokenize = input('\nTokenize the data? (y/n): ')

# Define maximum number of words to consider as features
max_features = 10000
# Define maximum length of a sequence
max_length = 200

# Check if tokenizer is saved already, if not create one
if tokenize == 'n' or tokenize == '':
    with open('RNN_Model/tokenizer.pkl', 'rb') as handle:
        tokenizer = pickle.load(handle)
    data = np.load('RNN_Model/tokenized_data.npy')

elif tokenize == 'y':

    print('\nTokenizing the data...')

    start = time.time()

    tokenizer = Tokenizer(num_words=max_features)
    df['text'] = df['text'].fillna('')
    tokenizer.fit_on_texts(df['text'])
    sequences = tokenizer.texts_to_sequences(df['text'])
    with open('RNN_Model/tokenizer.pkl', 'wb') as handle:
        pickle.dump(tokenizer, handle)
    data = pad_sequences(sequences, maxlen=max_length)
    np.save('RNN_Model/tokenized_data.npy', data)

    end = time.time()

    print('\nData tokenized.')
    print(f'\nTime elapsed: {(end-start):.2f} seconds')

word_index = tokenizer.word_index
print('Found %s unique tokens.' % len(word_index))


labels = df['label'].values

# Split the data into a training set and a validation set
X_train, X_val, y_train, y_val = train_test_split(data, labels, test_size=0.2, random_state=42)

############################################# Train the model #############################################

train = input('\nTrain the model? (y/n): ')

if train == 'y':

    model = Sequential()
    model.add(Embedding(max_features, 32))  # Reduced number of units in the Embedding layer
    model.add(LSTM(32, dropout=0.2, recurrent_dropout=0.2))  # Reduced number of units in the LSTM layer
    model.add(Dense(1, activation='sigmoid', kernel_regularizer=regularizers.l2(0.01)))  # Added L2 regularization to the Dense layer

    batch_size = 256 * 8
    epochs = 20

    optimize = keras.optimizers.legacy.Adam(lr = 0.001)

    model.compile(loss='binary_crossentropy',
                optimizer=optimize,
                metrics=['accuracy'])

    print(model.summary())

    # Specify the path where you want to save the model
    filepath = "RNN_Model/saved_model.h5"

    # Initialize ModelCheckpoint
    checkpoint = ModelCheckpoint(filepath, monitor='val_loss', verbose=1, save_best_only=True, mode='min')

    # Pass the ModelCheckpoint callback to model.fit() function
    history = model.fit(X_train, y_train, 
                        batch_size=batch_size, 
                        epochs=epochs, 
                        validation_data=(X_val, y_val),
                        callbacks=[checkpoint])

    plot = input('\nPlot the loss? (y/n): ')

    if plot == 'y':

        fig, ax = plt.subplots(figsize=(7,4))
        ax.plot(history.history['loss'], label='Train', color = 'k')
        ax.plot(history.history['val_loss'], label='Validation', color = 'b')
        ax.set_title('Model Loss vs Epoch')
        ax.set_ylabel('Loss')
        ax.set_xlabel('Epoch')
        ax.set_xticks(np.arange(0, epochs+1, 5))
        ax.legend(loc='upper right')
        plt.tight_layout()
        plt.savefig('RNN_Model/loss_saved.png', dpi=300, bbox_inches='tight')
        plt.show()
        save = input('\nSave the plot? (y/n): ')
        if save == 'n' or '':
            os.remove('RNN_Model/loss_saved.png')


if train == 'n' or train == '':
    #name = input('\nName of the model: ')
    model = load_model(f'RNN_Model/saved_model.h5')

############################################# Evaluate the model #############################################

print('\nEvaluating the model...')

def evaluate_model(model, X_test, y_test):
    # Calculate loss and accuracy 
    loss, accuracy = model.evaluate(X_test, y_test, verbose=0)
    print("Loss: ", loss)
    print("Accuracy: ", accuracy)

    # Generate predictions
    y_pre = model.predict(X_test)

    # Choose a threshold for classification
    y_pred = np.where(y_pre >= 0.5, 1, 0)

    # Calculate Confusion Matrix
    print("Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred))

    # Calculate Precision, Recall, F1 Score
    print("Classification Report:")
    print(classification_report(y_test, y_pred))

    # Calculate ROC Curve and AUC
    fpr, tpr, thresholds = roc_curve(y_test, y_pred)
    roc_auc = auc(fpr, tpr)
    print("AUC of ROC Curve:", roc_auc)
    
    # Return all metrics
    return loss, accuracy, y_pre, y_pred, roc_auc

loss, accuracy, y_pre, y_pred, roc_auc = evaluate_model(model, X_val, y_val)

# plot_eval = input('\nPlot the evaluation? (y/n): ')

# def plot_confusion_matrix_and_roc(model, X_test, y_test, y_pre):
#     # Get the model predictions
#     #y_pred = model.predict(X_test)
#     y_pred_r = np.round(y_pre).flatten()  # convert predictions to 0/1

#     # Compute the confusion matrix
#     conf_matrix = confusion_matrix(y_test, y_pred_r)

#     # Compute ROC curve and ROC area
#     fpr, tpr, _ = roc_curve(y_test, y_pre)
#     roc_auc = auc(fpr, tpr)

#     # Create a figure
#     fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

#     # Plot confusion matrix
#     sns.heatmap(conf_matrix, cmap='Blues', annot=True, fmt="d", ax=ax1)
#     ax1.set_title('Confusion Matrix')
#     ax1.set_xlabel('Predicted Label')
#     ax1.set_ylabel('True Label')

#     # Plot ROC curve
#     ax2.plot(fpr, tpr, color='b', label='ROC curve (area = %0.2f)' % roc_auc)
#     ax2.plot([0, 1], [0, 1], color='k', linestyle='--')
#     ax2.set_xlim([-0.01, 1.01])
#     ax2.set_ylim([-0.01, 1.01])
#     ax2.set_xlabel('False Positive Rate')
#     ax2.set_ylabel('True Positive Rate')
#     ax2.set_title('Receiver Operating Characteristic')
#     legend(ax2,loc='lower right')
#     plot_params(ax2)

#     # Save the figure
#     plt.savefig('Plots/evaluation.png', dpi=300, bbox_inches='tight')

#     plt.show()

# if plot_eval == 'y' or plot_eval == '':

#     plot_confusion_matrix_and_roc(model, X_val, y_val, y_pre)

#     save = input('\nSave the plot? (y/n): ')
#     if save == 'n' or '':
#         os.remove('Plots/evaluation.png')




# # Reduce complexity of the model and retrain, so number of parameters decreases. This may help with overfitting.

def RNN_LSTM(X, y, epochs, batchsize, name = 1):
    import pandas as pd
    import numpy as np
    from tensorflow import keras
    from keras.preprocessing.text import Tokenizer
    from keras.utils import pad_sequences
    from keras.models import Sequential
    from keras.layers import Dense, Embedding, LSTM, SpatialDropout1D, BatchNormalization
    from sklearn.model_selection import train_test_split
    from keras.utils.np_utils import to_categorical
    from keras.callbacks import EarlyStopping
    from keras.layers import Dropout
    from keras.callbacks import LearningRateScheduler
    import math
    from keras.models import load_model
    from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc
    from keras.callbacks import ModelCheckpoint
    import seaborn as sns
    from keras import regularizers

    import re
    import os

    import time
    import sys
    import pickle
    import matplotlib.pyplot as plt

    font = {'family' : 'monospace'}
    plt.rc('font', **font)

    ############################################# Tokenize Data #############################################

    tokenize = input('\nTokenize the data? (y/n): ')

    # Define maximum number of words to consider as features
    max_features = 10000
    # Define maximum length of a sequence
    max_length = 200

    # Check if tokenizer is saved already, if not create one
    if tokenize == 'n' or tokenize == '':
        with open(f'RNN_Model/tokenizer_{name}.pkl', 'rb') as handle:
            tokenizer = pickle.load(handle)
        data = np.load(f'RNN_Model/tokenized_data_{name}.npy')

    elif tokenize == 'y':

        print('\nTokenizing the data...')

        start = time.time()

        tokenizer = Tokenizer(num_words=max_features)
        df['text'] = df['text'].fillna('')
        tokenizer.fit_on_texts(df['text'])
        sequences = tokenizer.texts_to_sequences(df['text'])
        with open(f'RNN_Model/tokenizer_{name}.pkl', 'wb') as handle:
            pickle.dump(tokenizer, handle)
        data = pad_sequences(sequences, maxlen=max_length)
        np.save(f'RNN_Model/tokenized_data_{name}.npy', data)

        end = time.time()

        print('\nData tokenized.')
        print(f'\nTime elapsed: {(end-start):.2f} seconds')

    word_index = tokenizer.word_index
    print('Found %s unique tokens.' % len(word_index))

    # Split the data into a training set and a validation set
    X_train, X_val, y_train, y_val = train_test_split(data, labels, test_size=0.2, random_state=42)

    ############################################# Train the model #############################################

    train = input('\nTrain the model? (y/n): ')

    if train == 'y':

        model = Sequential()
        model.add(Embedding(max_features, 32))  # Reduced number of units in the Embedding layer
        model.add(LSTM(32, dropout=0.2, recurrent_dropout=0.2))  # Reduced number of units in the LSTM layer
        model.add(Dense(1, activation='sigmoid', kernel_regularizer=regularizers.l2(0.01)))  # Added L2 regularization to the Dense layer

        batch_size = 256 * 8
        epochs = 20

        optimize = keras.optimizers.legacy.Adam(lr = 0.001)

        model.compile(loss='binary_crossentropy',
                    optimizer=optimize,
                    metrics=['accuracy'])

        print(model.summary())

        # Specify the path where you want to save the model
        filepath = f"RNN_Model/saved_model_{name}.h5"

        # Initialize ModelCheckpoint
        checkpoint = ModelCheckpoint(filepath, monitor='val_loss', verbose=1, save_best_only=True, mode='min')

        # Pass the ModelCheckpoint callback to model.fit() function
        history = model.fit(X_train, y_train, 
                            batch_size=batch_size, 
                            epochs=epochs, 
                            validation_data=(X_val, y_val),
                            callbacks=[checkpoint])

        plot = input('\nPlot the loss? (y/n): ')

        if plot == 'y':

            fig, ax = plt.subplots(figsize=(7,4))
            ax.plot(history.history['loss'], label='Train', color = 'k')
            ax.plot(history.history['val_loss'], label='Validation', color = 'b')
            ax.set_title('Model Loss vs Epoch')
            ax.set_ylabel('Loss')
            ax.set_xlabel('Epoch')
            ax.set_xticks(np.arange(0, epochs+1, 5))
            ax.legend(loc='upper right')
            plt.tight_layout()
            plt.savefig(f'RNN_Model/loss_saved_{name}.png', dpi=300, bbox_inches='tight')
            plt.show()
            save = input('\nSave the plot? (y/n): ')
            if save == 'n' or '':
                os.remove(f'RNN_Model/loss_saved_{name}.png')


    if train == 'n' or train == '':
        #name = input('\nName of the model: ')
        model = load_model(f'RNN_Model/saved_model_{name}.h5')

    ############################################# Evaluate the model #############################################

    print('\nEvaluating the model...')

    def evaluate_model(model, X_test, y_test):
        # Calculate loss and accuracy 
        loss, accuracy = model.evaluate(X_test, y_test, verbose=0)
        print("Loss: ", loss)
        print("Accuracy: ", accuracy)

        # Generate predictions
        y_pre = model.predict(X_test)

        # Choose a threshold for classification
        y_pred = np.where(y_pre >= 0.5, 1, 0)

        # Calculate Confusion Matrix
        print("Confusion Matrix:")
        print(confusion_matrix(y_test, y_pred))

        # Calculate Precision, Recall, F1 Score
        print("Classification Report:")
        print(classification_report(y_test, y_pred))

        # Calculate ROC Curve and AUC
        fpr, tpr, thresholds = roc_curve(y_test, y_pred)
        roc_auc = auc(fpr, tpr)
        print("AUC of ROC Curve:", roc_auc)
        
        # Return all metrics
        return loss, accuracy, y_pre, y_pred, roc_auc

    loss, accuracy, y_pre, y_pred, roc_auc = evaluate_model(model, X_val, y_val)