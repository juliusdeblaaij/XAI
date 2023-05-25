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
from nltk.metrics import ConfusionMatrix
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt
import time
import sys,os

# Load the files, including features, images and labels. 

import inspect
import os
dirname = os.path.dirname(os.path.abspath(inspect.stack()[0][1]))
X_train_file_path = os.path.join(dirname, 'data_df_X_train.csv')
y_train_file_path = os.path.join(dirname, 'data_df_y_train.csv')
X_test_file_path = os.path.join(dirname, 'data_df_X_test.csv')
y_test_file_path = os.path.join(dirname, 'data_df_y_test.csv')

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
    data = {'Images': y_train_images, 'Features': X_train, 'Labels': y_train_labels}

    mode = 'Learning'

    return xDNN(data, mode)


def validate(training_results):
    data = {'xDNNParms': training_results['xDNNParms'], 'Images': y_test_images, 'Features': X_test,
             'Labels': y_test_labels}

    mode = 'Validation'
    return xDNN(data, mode)


def classify(training_results, features):
    data = {'xDNNParms': training_results['xDNNParms'], 'Features': features}

    mode = 'Classify'
    return xDNN(data, mode)


if __name__ == "__main__":
    startTraining = time.time()
    training_results = train()
    endTraining = time.time()

    print("Model trained in: ", round(endTraining - startTraining, 2), "seconds")

    startValidation = time.time()
    validation_results = validate(training_results)
    endValidation = time.time()

    # Elapsed Time
    print("Time: ", round(endValidation - startValidation, 2), "seconds")
    # accuracy: (tp + tn) / (p + n)
    accuracy = accuracy_score(y_test_labels, validation_results['EstLabs'])
    print('Accuracy: %f' % accuracy)
    # precision tp / (tp + fp)
    precision = precision_score(y_test_labels, validation_results['EstLabs'], average='micro')
    print('Precision: %f' % precision)
    # recall: tp / (tp + fn)
    recall = recall_score(y_test_labels, validation_results['EstLabs'], average='micro')
    print('Recall: %f' % recall)
    # f1: 2 tp / (2 tp + fp + fn)
    f1 = f1_score(y_test_labels, validation_results['EstLabs'], average='micro')
    print('F1 score: %f' % f1)
    # kappa
    kappa = cohen_kappa_score(y_test_labels, validation_results['EstLabs'])
    print('Cohens kappa: %f' % kappa)

    y_test_labels = list(y_test_labels.flatten())
    estimated_labels = validation_results['EstLabs'].flatten().astype(int)

    y_test_labels_reclassified = []

    for label in y_test_labels:
        if label == 0:
            y_test_labels_reclassified.append(0)
        elif label == 1:
            y_test_labels_reclassified.append(1)
        elif label == 2:
            y_test_labels_reclassified.append(2)
        elif label == 3:
            y_test_labels_reclassified.append(3)
        elif label == 4:
            y_test_labels_reclassified.append(5)
        elif label == 5:
            y_test_labels_reclassified.append(8)

    estimated_labels_reclassified = []

    for label in estimated_labels:
        if label == 0:
            estimated_labels_reclassified.append(0)
        elif label == 1:
            estimated_labels_reclassified.append(1)
        elif label == 2:
            estimated_labels_reclassified.append(2)
        elif label == 3:
            estimated_labels_reclassified.append(3)
        elif label == 4:
            estimated_labels_reclassified.append(5)
        elif label == 5:
            estimated_labels_reclassified.append(8)


    # confusion matrix
    matrix = ConfusionMatrix(y_test_labels_reclassified, estimated_labels_reclassified)
    print("Confusion Matrix:")
    print(matrix)
