
def RNN_LSTM(df, labels, epochs, batchsize, name = 1):
    """
    Function to run a RNN-LSTM model on the data.
    X: dataframe with columns 'title', 'text', 'label' and 'total' (Can also just be 'text' and 'label', or 'text' and 'title')
    y: labels
    epochs: number of epochs to train the model
    batchsize: batch size to train the model 
    name: name of the model ( used to save the model )
    """

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
    import time
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