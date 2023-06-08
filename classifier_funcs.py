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
        model.add(Embedding(max_features+1, 32))  # Reduced number of units in the Embedding layer
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


        model_best = load_model(f'RNN_Model/saved_model_{name}.h5')

        
    if train == 'n' or train == '':
        name_in = input('\nName of the model: ')
        model_best = load_model(f'RNN_Model/saved_model_{name_in}.h5')

    ############################################# Evaluate the model #############################################

    print('\nEvaluating the model...')

    loss, accuracy, y_pre, y_pred, roc_auc = evaluate_model(model_best, X_val, y_val)


from lightgbm import LGBMClassifier
from lightgbm import early_stopping, log_evaluation
from sklearn.metrics import accuracy_score, log_loss, confusion_matrix, roc_auc_score

# def LightGBM(X_train, y_train, X_test, y_test):
#     """
#     Trains a LightGBM binary classifier on the provided training data, 
#     then makes predictions on the test data and evaluates the model's performance.

#     Parameters
#     ----------
#     X_train : array-like of shape (n_samples, n_features)
#         The training input samples.
#     y_train : array-like of shape (n_samples,)
#         The target values (class labels) as integers or strings.
#     X_test : array-like of shape (n_samples, n_features)
#         The test input samples.
#     y_test : array-like of shape (n_samples,)
#         The true values (class labels) for the test input samples.

#     Returns
#     -------
#     accuracy : float
#         The accuracy of the model on the test data.
#     bce_loss : float
#         The binary cross-entropy loss of the model on the test data.
#     conf_matrix : ndarray of shape (n_classes, n_classes)
#         The confusion matrix of the model's predictions on the test data.
#     roc_auc : float
#         The ROC AUC score of the model on the test data.
#     """

#     print("\nTraining LightGBM classifier...")

#     # Initialize our classifier
#     clf = LGBMClassifier()

#     # Train the classifier
#     clf.fit(X_train, y_train)

#     # Make predictions on the test set
#     y_pred = clf.predict(X_test)

#     # Predict probabilities for log loss
#     y_pred_prob = clf.predict_proba(X_test)

#     print("\nEvaluating model...")

#     # Calculate accuracy
#     accuracy = accuracy_score(y_test, y_pred)
#     print(f"\nAccuracy: {accuracy}")

#     # Calculate log loss (binary cross entropy)
#     bce_loss = log_loss(y_test, y_pred_prob)
#     print(f"\nBinary Cross Entropy Loss: {bce_loss}")

#     # Calculate confusion matrix
#     conf_matrix = confusion_matrix(y_test, y_pred)
#     print(f"\nConfusion Matrix: \n{conf_matrix}")

#     # Calculate ROC AUC score
#     roc_auc = roc_auc_score(y_test, y_pred_prob[:,1])
#     print(f"\nROC AUC Score: {roc_auc}")

#     return accuracy, bce_loss, conf_matrix, roc_auc

def LightGBM(X_train, y_train, X_test, y_test, early_stop = True):
    """
    Trains a LightGBM binary classifier on the provided training data, 
    then makes predictions on the test data and evaluates the model's performance.

    Parameters
    ----------
    X_train : array-like of shape (n_samples, n_features)
        The training input samples.
    y_train : array-like of shape (n_samples,)
        The target values (class labels) as integers or strings.
    X_test : array-like of shape (n_samples, n_features)
        The test input samples.
    y_test : array-like of shape (n_samples,)
        The true values (class labels) for the test input samples.

    Returns
    -------
    accuracy : float
        The accuracy of the model on the test data.
    bce_loss : float
        The binary cross-entropy loss of the model on the test data.
    conf_matrix : ndarray of shape (n_classes, n_classes)
        The confusion matrix of the model's predictions on the test data.
    roc_auc : float
        The ROC AUC score of the model on the test data.
    """

    print("\nTraining LightGBM classifier...")

    # Initialize our classifier with specified parameters to combat overfitting
    clf = LGBMClassifier(n_estimators=100, learning_rate=0.01, max_depth=3, subsample=0.8, reg_alpha=0.1, reg_lambda=0.1)

    if early_stop == True:
        # Further split the training data into training and validation sets for early stopping
        X_train_sub, X_val, y_train_sub, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42)

        # Train the classifier with early stopping
        clf.fit(X_train_sub, y_train_sub, eval_set=[(X_val, y_val)], callbacks=[early_stopping(50), log_evaluation(period=100)])

    else:
        # Train the classifier
        clf.fit(X_train, y_train)

    # Make predictions on the test set
    y_pred = clf.predict(X_test)

    # Predict probabilities for log loss
    y_pred_prob = clf.predict_proba(X_test)

    print("\nEvaluating model...")

    # Calculate accuracy
    accuracy = accuracy_score(y_test, y_pred)
    print(f"\nAccuracy: {accuracy}")

    # Calculate log loss (binary cross entropy)
    bce_loss = log_loss(y_test, y_pred_prob)
    print(f"\nBinary Cross Entropy Loss: {bce_loss}")

    # Calculate confusion matrix
    conf_matrix = confusion_matrix(y_test, y_pred)
    print(f"\nConfusion Matrix: \n{conf_matrix}")

    # Calculate ROC AUC score
    roc_auc = roc_auc_score(y_test, y_pred_prob[:,1])
    print(f"\nROC AUC Score: {roc_auc}")

    return accuracy, bce_loss, conf_matrix, roc_auc