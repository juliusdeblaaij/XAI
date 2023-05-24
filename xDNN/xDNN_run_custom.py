#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Please cite:
Angelov, P., & Soares, E. (2020). Towards explainable deep neural networks (xDNN). Neural Networks.

"""

###############################################################################
import pandas as pd

from xDNN.xDNN_class import *
from numpy import genfromtxt
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.metrics import cohen_kappa_score
from sklearn.metrics import roc_auc_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt
import time
from doc2vec import doc2vec
import doxpy
# Load the files, including features, images and labels.

X_train_file_path = r'C:/Users/SKIKK/PycharmProjects/XAI/data_df_X_train.csv'
y_train_file_path = r'C:/Users/SKIKK/PycharmProjects/XAI/data_df_y_train.csv'
X_test_file_path = r'C:/Users/SKIKK/PycharmProjects/XAI/data_df_X_test.csv'
y_test_file_path = r'C:/Users/SKIKK/PycharmProjects/XAI/data_df_y_test.csv'

X_train = genfromtxt(X_train_file_path, delimiter=',')
y_train = pd.read_csv(y_train_file_path, delimiter=',', header=None)
X_test = genfromtxt(X_test_file_path, delimiter=',')
y_test = pd.read_csv(y_test_file_path, delimiter=',', header=None)

# Print the shape of the data

print("###################### Data Loaded ######################")
print("Data Shape:   ")
print("X train: ", X_train.shape)
print("Y train: ", y_train.shape)
print("X test: ", X_test.shape)
print("Y test: ", y_test.shape)

y_train_labels = []

for label in y_train[1]:
    if label == ' ':
        y_train_labels.append(0)
        continue

    y_train_labels.append(int(label))

pd_y_train_labels = pd.DataFrame(y_train_labels)
pd_y_train_images = y_train[0]

y_test_labels = []

for label in y_test[1]:
    if label == ' ':
        y_test_labels.append(0)
        continue
    y_test_labels.append(int(label))

pd_y_test_labels = pd.DataFrame(y_test_labels)
pd_y_test_images = y_test[0]

# Convert Pandas to Numpy
y_train_labels = pd_y_train_labels.to_numpy().flatten()
y_train_images = pd_y_train_images.to_numpy()

y_test_labels = pd_y_test_labels.to_numpy()
y_test_images = pd_y_test_images.to_numpy()


def train():
    # Model Learning
    Input1 = {}

    Input1['Images'] = y_train_images
    Input1['Features'] = X_train
    Input1['Labels'] = y_train_labels

    Mode1 = 'Learning'

    return xDNN(Input1,Mode1)


###############################################################################

# Load the files, including features, images and labels for the validation mode

def validate(training_results):
    Input2 = {}
    Input2['xDNNParms'] = training_results['xDNNParms']
    Input2['Images'] = y_test_images
    Input2['Features'] = X_test
    Input2['Labels'] = y_test_labels
    Mode2 = 'Validation'
    return xDNN(Input2, Mode2)


def classify(training_results, features):
    classification_input = {}
    classification_input['xDNNParms'] = training_results['xDNNParms']
    classification_input['Features'] = features
    classification_mode = 'classify'

    return xDNN(classification_input, classification_mode)


if __name__ == "__main__":
    start = time.time()
    Output1 = train()

    end = time.time()

    print("###################### Model Trained ####################")

    print("Time: ", round(end - start, 2), "seconds")

    startValidation = time.time()
    Output2 = validate(Output1)
    endValidation = time.time()

    print("###################### Results ##########################")

    # Elapsed Time
    print("Time: ", round(endValidation - startValidation, 2), "seconds")
    # accuracy: (tp + tn) / (p + n)
    accuracy = accuracy_score(y_test_labels, Output2['EstLabs'])
    print('Accuracy: %f' % accuracy)
    # precision tp / (tp + fp)
    precision = precision_score(y_test_labels, Output2['EstLabs'], average='micro')
    print('Precision: %f' % precision)
    # recall: tp / (tp + fn)
    recall = recall_score(y_test_labels, Output2['EstLabs'], average='micro')
    print('Recall: %f' % recall)
    # f1: 2 tp / (2 tp + fp + fn)
    f1 = f1_score(y_test_labels, Output2['EstLabs'], average='micro')
    print('F1 score: %f' % f1)
    # kappa
    kappa = cohen_kappa_score(y_test_labels, Output2['EstLabs'])
    print('Cohens kappa: %f' % kappa)

    # confusion matrix
    matrix = confusion_matrix(y_test_labels, Output2['EstLabs'])
    print("Confusion Matrix: ", matrix)

