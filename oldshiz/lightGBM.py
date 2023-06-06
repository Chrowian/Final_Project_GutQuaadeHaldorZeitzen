from sklearn.model_selection import train_test_split
from lightgbm import LGBMClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, roc_curve, auc
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import log_loss
from scipy.sparse import hstack

import re
import os
import pandas as pd
import seaborn as sns

import time
import sys
import pickle
import matplotlib.pyplot as plt

font = {'family' : 'monospace'}

# Load the data
print('\nReading in the data...')

df = pd.read_csv('WELFake_Dataset.csv')

print('\nData loaded.')

print('\nPreprocessing & Vectorising the data...')

start = time.time()

# Replace NaNs with empty string in 'title' and 'text' columns
df['title'] = df['title'].fillna('')
df['text'] = df['text'].fillna('')

# Initialize the TfidfVectorizer
vectorizer_title = TfidfVectorizer(max_features=5000, stop_words='english')
vectorizer_text = TfidfVectorizer(max_features=5000, stop_words='english')

# Fit and transform the vectorizer on our text
X_title = vectorizer_title.fit_transform(df['title'])
X_text = vectorizer_text.fit_transform(df['text'])

# Combine the features
X = hstack([X_title, X_text])

# Get the labels
y = df['label']

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

end = time.time()
print(f'\nData preprocessed & vectorized in {(end-start):.2f} seconds.')

print('\nTraining the model...')
start = time.time()

# Initialize our classifier
clf = LGBMClassifier()

# Train the classifier
clf.fit(X_train, y_train)

# Evaluate the model on the test set
score = clf.score(X_test, y_test)

y_pred_proba = clf.predict_proba(X_test)

# The probabilities of the positive class is at index 1
y_pred_proba_pos = y_pred_proba[:, 1]

# Compute log loss (Binary Cross Entropy)
loss = log_loss(y_test, y_pred_proba_pos)

end = time.time()
print(f'\nModel trained in {(end-start):.2f} seconds.')

print("Accuracy Score: %.3f%%" % (score * 100.0),'\n')
print("Binary Cross Entropy (Log Loss): %.3f" % loss,'\n')

y_pred = clf.predict(X_test)

# Evaluate the model
print(f'Accuracy: {(accuracy_score(y_test, y_pred)):.3f}')
print("\nConfusion Matrix: \n", confusion_matrix(y_test, y_pred))
print("\nClassification Report: \n", classification_report(y_test, y_pred))

# # Plot the confusion matrix and ROC curve
# def plot_confusion_matrix_and_roc(model, X_test, y_test, y_pred = None):
#     # Get the model predictions
#     #y_pred = model.predict(X_test)
#     #y_pred_r = np.round(y_pre).flatten()  # convert predictions to 0/1

#     # Compute the confusion matrix
#     conf_matrix = confusion_matrix(y_test, y_pred)

#     # Compute ROC curve and ROC area
#     fpr, tpr, _ = roc_curve(y_test, y_pred)
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
#     ax2.legend(ax2,loc='lower right')
#     plot_params(ax2)

#     # Save the figure
#     #plt.savefig('Plots/evaluation.png', dpi=300, bbox_inches='tight')

#     plt.show()

# #plot_confusion_matrix_and_roc(clf, X_test, y_test, y_pred)