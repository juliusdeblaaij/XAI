#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Please cite:
Angelov, P., & Soares, E. (2020). Towards explainable deep neural networks (xDNN). Neural Networks.

"""

import pandas as pd

from xDNN.xDNN_class import *

class RunxDNN:
    training_results = None

    def train(self, cases, features, labels):

        data = {'Images': cases, 'Features': features, 'Labels': labels}

        mode = 'Learning'

        training_results = xDNN(data, mode)
        return training_results

    def validate(self, training_results, test_features, test_cases):
        data = {'xDNNParms': training_results['xDNNParms'], 'Images': test_cases, 'Features': test_features,
                'Labels': test_cases}

        mode = 'Validation'
        return xDNN(data, mode)

    def classify(self, training_results, features):
        data = {'xDNNParms': training_results['xDNNParms'], 'Features': features}

        mode = 'Classify'
        return xDNN(data, mode)

"""
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
"""
