# Classifier Functions for the project
# Rune Zeitzen
# June 2023

############################################# Import packages #############################################

import pandas as pd
import numpy as np
from tensorflow import keras
from keras.preprocessing.text import Tokenizer
from keras.utils import pad_sequences
from keras.models import Sequential
from keras.layers import Dense, Embedding, LSTM, SpatialDropout1D, BatchNormalization
from sklearn.model_selection import train_test_split

from keras.models import load_model
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc
from keras.callbacks import ModelCheckpoint
from keras import regularizers
import os
import re
import time
import pickle
import matplotlib.pyplot as plt

############################################# Helper Functions #############################################

def get_next_model_number(directory):
    # Get list of all files/directories in the given directory
    files = os.listdir(directory)
    
    # Initialize the highest model number found to 0
    highest_num = 0

    # Regular expression to match 'saved_model_{number}'
    pattern = re.compile(r'saved_model_(\d+)\.h5')
    
    # Iterate through all files/directories
    for file in files:
        match = pattern.match(file)
        
        # If this file/directory matches the pattern
        if match:
            # Extract the number from the filename
            num = int(match.group(1))
            
            # If this number is higher than any we've seen before, update highest_num
            if num > highest_num:
                highest_num = num

    # Return the next model number
    return highest_num + 1

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


############################################# RNN-LSTM Function #############################################

def RNN_LSTM(data, labels, epoch = 20, batchsize = 256 * 8, max_features = 10000):
    """
    Function to run a RNN-LSTM model on the data.
    data: dataframe with columns 'title', 'text' and 'label'
    lables: labels for the data
    epochs: number of epochs to train the model
    batchsize: batch size to train the model 
    name: name of the model ( used to save the model )
    """

    name = get_next_model_number('RNN_Model')

    X_train, X_val, y_train, y_val = train_test_split(data, labels, test_size=0.2, random_state=42)

    ############################################# Train the model #############################################

    train = input('\nTrain the model? (y/n): ')

    if train == 'y':

        model = Sequential()
        model.add(Embedding(max_features, 32))  # Reduced number of units in the Embedding layer
        model.add(LSTM(32, dropout=0.2, recurrent_dropout=0.2))  # Reduced number of units in the LSTM layer
        model.add(Dense(1, activation='sigmoid', kernel_regularizer=regularizers.l2(0.01)))  # Added L2 regularization to the Dense layer

        batch_size = batchsize
        epochs = epoch

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

    loss, accuracy, y_pre, y_pred, roc_auc = evaluate_model(model, X_val, y_val)