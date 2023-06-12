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
import seaborn as sns




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

def evaluate_model(model, X_test, y_test, plot = False, vocab_size = None, name = None):
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
        conf_matrix = confusion_matrix(y_test, y_pred)
        print(conf_matrix)

        # Calculate Precision, Recall, F1 Score
        print("Classification Report:")
        print(classification_report(y_test, y_pred))

        # Calculate ROC Curve and AUC
        fpr, tpr, thresholds = roc_curve(y_test, y_pre)
        roc_auc = auc(fpr, tpr)
        print("AUC of ROC Curve:", roc_auc)

        if plot:
            # Create a figure
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

            fig.suptitle(f'Confusion Matrix and ROC Curve\nBest LSTM Model', fontsize=20, y = 1.05)

            # Plot confusion matrix
            sns.heatmap(conf_matrix, cmap='Blues', annot=True, fmt="d", ax=ax1, vmin = 0)
            #ax1.set_title('Confusion Matrix')
            ax1.set_xlabel('Predicted Label')
            ax1.set_ylabel('True Label')

            # Plot ROC curve
            ax2.plot(fpr, tpr, color='b', label='ROC curve (area = %0.2f)' % roc_auc)
            ax2.plot([0, 1], [0, 1], color='k', linestyle='--')
            ax2.set_xlim([-0.01, 1.01])
            ax2.set_ylim([-0.01, 1.01])
            ax2.set_xlabel('False Positive Rate')
            ax2.set_ylabel('True Positive Rate')
            plot_params(ax2)

            data = [accuracy, loss, roc_auc, vocab_size]
            
            string = f'Accuracy: {data[0]:.4f}\nBCE (LogLoss): {data[1]:.4f}\nROC AUC: {data[2]:.3f}\n\nVocabulary Size: {data[3]}'
            plot_text(ax2, 0.5, 0.1, string, size = 12, box = True, color = 'k', font_style = 'italic')

            # Save the figure
            plt.savefig(f'Result_Models/RNN/evaluation_RNNLSTM_{name}_{vocab_size}.png', dpi=300, bbox_inches='tight')

            plt.show()

        # Return all metrics
        return loss, accuracy, y_pre, y_pred, roc_auc


############################################# RNN-LSTM Function #############################################

def RNN_LSTM(data, labels, epoch = 20, batchsize = 256 * 8, max_features = 10000, names = None, plotIT = False, train = None):
    """
    Function to run a RNN-LSTM model on the data.
    data: dataframe with columns 'title', 'text' and 'label'
    lables: labels for the data
    epochs: number of epochs to train the model
    batchsize: batch size to train the model 
    name: name of the model ( used to save the model )
    """

    name = get_next_model_number('Result_Models/RNN_MODELS')

    X_train, X_val, y_train, y_val = train_test_split(data, labels, test_size=0.3, random_state=42)

    X_val, X_test, y_val, y_test = train_test_split(X_val, y_val, test_size=0.5, random_state=42)

    ############################################# Train the model #############################################

    if train is None:

        train = input('\nTrain the model? (y/n): ')

    if train == 'y':

        model = Sequential()
        model.add(Embedding(max_features, 32)) 
        model.add(LSTM(32, dropout=0.2, recurrent_dropout=0.2)) 
        model.add(Dense(1, activation='sigmoid', kernel_regularizer=regularizers.l2(0.01))) 
        
        batch_size = batchsize
        epochs = epoch

        optimize = keras.optimizers.legacy.Adam(lr = 0.001)

        model.compile(loss='binary_crossentropy',
                    optimizer=optimize,
                    metrics=['accuracy'])

        print(model.summary())

        # Specify the path where you want to save the model
        filepath = f"Result_Models/RNN_MODELS/saved_model_{name}.h5"

        # Initialize ModelCheckpoint
        checkpoint = ModelCheckpoint(filepath, monitor='val_loss', verbose=1, save_best_only=True, mode='min')

        # Pass the ModelCheckpoint callback to model.fit() function
        history = model.fit(X_train, y_train, 
                            batch_size=batch_size, 
                            epochs=epochs, 
                            validation_data=(X_val, y_val),
                            callbacks=[checkpoint])
        
        if not plotIT:
            plot = input('\nPlot the loss? (y/n): ')

        else:
            plot = 'y'

        if plot == 'y':

            fig, ax = plt.subplots(figsize=(7,4))
            ax.plot(history.history['loss'], label='Train', color = 'k')
            ax.plot(history.history['val_loss'], label='Validation', color = 'b')
            ax.set_title('Model Loss vs Epoch')
            ax.set_ylabel('Loss')
            ax.set_xlabel('Epoch')
            ax.set_xticks(np.arange(0, epochs+1, 5))
            legend(ax, loc='upper right')
            plot_params(ax)
            plt.tight_layout()
            plt.savefig(f'Result_Models/RNN/loss_{names}_{max_features}.png', dpi=300, bbox_inches='tight')
            plt.show()

            if not plotIT:
                save = input('\nSave the plot? (y/n): ')

            else:
                save = 'y'
            if save == 'n' or '':
                os.remove(f'Result_Models/RNN/loss_{names}_{max_features}.png')


        model_best = load_model(f'Result_Models/RNN_MODELS/saved_model_{name}.h5')

        
    if train == 'n' or train == '':
        name_in = input('\nName of the model: ')
        model_best = load_model(f'Result_Models/RNN_MODELS/saved_model_{name_in}.h5')

    ############################################# Evaluate the model #############################################

    print('\nEvaluating the model...')

    loss, accuracy, y_pre, y_pred, roc_auc = evaluate_model(model_best, X_test, y_test, plot = plotIT, vocab_size=max_features, name = names)


from lightgbm import LGBMClassifier
from lightgbm import early_stopping, log_evaluation
from sklearn.metrics import accuracy_score, log_loss, confusion_matrix, roc_auc_score

import matplotlib.pyplot as plt


def LightGBM(X_train, y_train, X_test, y_test):
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
    test_size : float, optional (default=0.5)
        The proportion of the test data to be used as the test set, with the remainder used as the validation set.
    random_state : int, optional (default=42)
        The seed used by the random number generator.
    plot : boolean, optional (default=False)
        If True, plot a subplot with a confusion matrix and a ROC curve.

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

    print("\nSplitting test data into validation and test sets...")

    X_val, X_test_new, y_val, y_test_new = train_test_split(X_test, y_test, test_size=0.5)

    print("\nTraining LightGBM classifier...")

    # Initialize our classifier with specified parameters to combat overfitting
    clf = LGBMClassifier(n_estimators=100, learning_rate=0.01, max_depth=3, subsample=0.8, reg_alpha=0.1, reg_lambda=0.1)

    # Train the classifier
    clf.fit(X_train, y_train, eval_set=(X_val, y_val), early_stopping_rounds=50, verbose=False)

    print("\nEvaluating model...")

    # Make predictions on the new test set
    y_pred = clf.predict(X_test_new)

    # Predict probabilities for log loss
    y_pred_prob = clf.predict_proba(X_test_new)

    # Calculate accuracy
    accuracy = accuracy_score(y_test_new, y_pred)
    print(f"\nAccuracy: {accuracy}")

    # Calculate log loss (binary cross entropy)
    bce_loss = log_loss(y_test_new, y_pred_prob)
    print(f"\nBinary Cross Entropy Loss: {bce_loss}")

    # Calculate confusion matrix
    conf_matrix = confusion_matrix(y_test_new, y_pred)
    print(f"\nConfusion Matrix: \n{conf_matrix}")

    # Calculate ROC AUC score
    roc_auc = roc_auc_score(y_test_new, y_pred_prob[:,1])
    print(f"\nROC AUC Score: {roc_auc}")

    return accuracy, bce_loss, conf_matrix, roc_auc, y_pred, y_pred_prob[:,1], clf, y_test_new


####################

def plot_params(ax, xlim = None, ylim = None, minor_grid = True, labsize = 12, xlocator = None, ylocator = None, log = (False,False)):

    ax.minorticks_on()
    ax.tick_params(axis='both', length = 10, which='major', labelsize=labsize)
    ax.tick_params(axis='both', length = 5, which='minor', labelsize=labsize)
    ax.grid(which = 'major',linestyle = ':',color = '0.25')
    
    if minor_grid:
        ax.grid(which = 'minor',linestyle = ':',color = '0.75')

    if xlim is not None:
        ax.set_xlim(xlim)

    if ylim is not None:
        ax.set_ylim(ylim)

    if xlocator is not None:
        from matplotlib.ticker import AutoMinorLocator
        minor_locator = AutoMinorLocator(xlocator)
        ax.xaxis.set_minor_locator(minor_locator)
    if ylocator is not None:
        from matplotlib.ticker import AutoMinorLocator
        minor_locator = AutoMinorLocator(ylocator)
        ax.yaxis.set_minor_locator(minor_locator)


    if log[0]:
        ax.set_xscale('log')
    if log[1]:
        ax.set_yscale('log')

####################

def legend(ax, loc = 'lower left', size = 10, cols = None, fancy = False, shad = True, title = None, titsize = 12, bbox_to_anchor = None):
    if bbox_to_anchor is None:
        if title is None:
            if cols is None:
                ax.legend(loc=loc,prop={'size': size},framealpha = 1, facecolor = '1',edgecolor = 'black', shadow = shad, fancybox = fancy)
            if cols is not None:
                ax.legend(loc=loc,ncol = cols,prop={'size': size},framealpha = 1, facecolor = '1',edgecolor = 'black', shadow = shad, fancybox = fancy)
    if bbox_to_anchor is not None:
        if title is not None:
            if cols is None:
                ax.legend(loc=loc,prop={'size': size},framealpha = 1, facecolor = '1',edgecolor = 'black', shadow = shad, fancybox = fancy, title = title, title_fontsize = titsize, bbox_to_anchor = bbox_to_anchor)
            if cols is not None:
                ax.legend(loc=loc,ncol = cols,prop={'size': size},framealpha = 1, facecolor = '1',edgecolor = 'black', shadow = shad, fancybox = fancy,title = title, title_fontsize = titsize, bbox_to_anchor = bbox_to_anchor)

####################

def plot_text(ax, x, y, string, size = 12, box = True, color = 'k', font_style = 'normal'):
    if box:
        return ax.text(x, y, string,c = color,size = size,bbox=dict(boxstyle='Square', facecolor='white', alpha=1), family='monospace', fontstyle = font_style)
    else:
        return ax.text(x, y, string,c = color,size = size, family='monospace', fontstyle = font_style)

# Plot the confusion matrix and ROC curve
def plot_confusion_matrix_and_roc(y_test, y_pred, y_pred_probs, name, vobab, type = 'LightGBM', data = None):

    font = {'family': 'monospace'}
    plt.rc('font', **font)

    # Compute the confusion matrix
    conf_matrix = confusion_matrix(y_test, y_pred)

    # Compute ROC curve and ROC area
    fpr, tpr, _ = roc_curve(y_test, y_pred_probs)
    roc_auc = auc(fpr, tpr)

    # Create a figure
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    fig.suptitle(f'Confusion Matrix and ROC Curve\nBest {type} Model', fontsize=20, y = 1.05)

    # Plot confusion matrix
    sns.heatmap(conf_matrix, cmap='Blues', annot=True, fmt="d", ax=ax1, vmin = 0)
    #ax1.set_title('Confusion Matrix')
    ax1.set_xlabel('Predicted Label')
    ax1.set_ylabel('True Label')

    # Plot ROC curve
    ax2.plot(fpr, tpr, color='b', label='ROC curve (area = %0.2f)' % roc_auc)
    ax2.plot([0, 1], [0, 1], color='k', linestyle='--')
    ax2.set_xlim([-0.01, 1.01])
    ax2.set_ylim([-0.01, 1.01])
    ax2.set_xlabel('False Positive Rate')
    ax2.set_ylabel('True Positive Rate')
    #ax2.set_title('Receiver Operating Characteristic')
    if data is None:
        legend(ax2,loc='lower right')
    plot_params(ax2)

    if data is not None:
        string = f'Accuracy: {data[0]:.4f}\nBCE (LogLoss): {data[1]:.4f}\nROC AUC: {data[2]:.3f}\n\nVocabulary Size: {data[3]}'
        plot_text(ax2, 0.5, 0.1, string, size = 12, box = True, color = 'k', font_style = 'italic')

    # Save the figure
    plt.savefig(f'Result_Models/Tree/evaluation_{type}_{name}_{vobab}.png', dpi=300, bbox_inches='tight')

    plt.show()

####################

import numpy as np
from sklearn.model_selection import learning_curve
import matplotlib.pyplot as plt
from lightgbm import LGBMClassifier

def plot_learning_curve(model, X, y, cv = 5, num = 10):


    train_sizes, train_scores, test_scores = learning_curve(model, X, y, cv=cv, train_sizes=np.linspace(0.1, 1.0, num))

    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)

    plt.fill_between(train_sizes, train_scores_mean - train_scores_std, train_scores_mean + train_scores_std, alpha=0.1, color='r')
    plt.fill_between(train_sizes, test_scores_mean - test_scores_std, test_scores_mean + test_scores_std, alpha=0.1, color='g')

    plt.plot(train_sizes, train_scores_mean, 'o-', color='r', label='Training score')
    plt.plot(train_sizes, test_scores_mean, 'o-', color='g', label='Cross-validation score')

    plt.title('Learning curves with error bars')
    plt.xlabel('Training examples')
    plt.ylabel('Score')
    plt.legend(loc='best')
    plt.show()

####################

from sklearn.model_selection import cross_val_score

def cross_val(model, X, y, cv = 5):
    scores = cross_val_score(model, X, y, cv=cv)

    print("Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))

    return scores