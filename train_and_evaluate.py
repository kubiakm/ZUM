import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import precision_recall_curve, auc
import matplotlib.pyplot as plt
import seaborn as sns
from test_functions import plot_PR_curve
from our_knn import *

def euclidean_distance(x1, x2):
    return np.sqrt(np.sum((x1 - x2) ** 2))

def manhattan_distance(x1, x2):
    return np.sum(np.abs(x1 - x2))

def minkowski_distance(x1, x2, p=2):
    return np.sum(np.abs(x1 - x2)**p)**(1/p)

def train_for_different_distance_func(X, y, distance_names):
    distance_funcs = {
        'euclidean_distance': euclidean_distance,
        'manhattan_distance': manhattan_distance,
        'minkowski_distance': lambda x1, x2: minkowski_distance(x1, x2, p=2)
    }

    results = {name: {'accuracies': [], 'scores': [], 'labels': []} for name in distance_names}
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    for name in distance_names:
        func = distance_funcs[name]
        detector = KNNAnomalyDetector(k=10, distance_func=func, knn_or_kth=1)
        for train_index, test_index in skf.split(X, y):
            X_train, X_test = X[train_index], X[test_index]
            y_train, y_test = y[train_index], y[test_index]

            detector.fit(X_train[y_train == 1])

            predictions = detector.predict(X_test, detector.threshold_)
            scores = detector.decision_function(X_test)
            scores = -scores
            precision, recall, _ = precision_recall_curve(y_test, scores)
            pr_auc = auc(recall, precision)

            results[name]['accuracies'].append(np.mean(predictions == y_test))
            results[name]['scores'].extend(scores)
            results[name]['labels'].extend(y_test)

    accuracies = [np.mean(results[name]['accuracies']) for name in distance_names]
    plt.figure(figsize=(10, 5))
    sns.barplot(x=distance_names, y=accuracies)
    plt.title('Accuracy by different distance metrics')
    plt.xlabel('Distance metrics')
    plt.ylabel('Accuracy')
    plt.show()

    plt.figure(figsize=(10, 8))
    for name in distance_names:
        scores = results[name]['scores']
        labels = results[name]['labels']
        plot_PR_curve(scores, labels, f'KNN Distance function={name}')
    plt.title('Precision-Recall curves')
    plt.show()
    
def train_for_different_ks(X, y, ks):
    results = {k: {'accuracies': [], 'scores': [], 'labels': []} for k in ks}
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    for k in ks:
        detector = KNNAnomalyDetector(k=k, distance_func=euclidean_distance, knn_or_kth=1)
        for train_index, test_index in skf.split(X, y):
            X_train, X_test = X[train_index], X[test_index]
            y_train, y_test = y[train_index], y[test_index]

            detector.fit(X_train[y_train == 1])

            predictions = detector.predict(X_test, detector.threshold_)
            scores = detector.decision_function(X_test)
            scores = -scores  
            precision, recall, _ = precision_recall_curve(y_test, scores)
            pr_auc = auc(recall, precision)

            results[k]['accuracies'].append(np.mean(predictions == y_test))
            results[k]['scores'].extend(scores)
            results[k]['labels'].extend(y_test)

    accuracies = [np.mean(results[k]['accuracies']) for k in ks]
    plt.figure(figsize=(10, 5))
    sns.barplot(x=ks, y=accuracies)
    plt.title('Accuracy by K values')
    plt.xlabel('K')
    plt.ylabel('Accuracy')
    plt.show()

    plt.figure(figsize=(10, 8))
    for k in results:
        scores = results[k]['scores']
        labels = results[k]['labels']
        plot_PR_curve(scores, labels, f'KNN K={k}')
    plt.title('Precision-Recall curves')
    plt.show()
    
def train_for_knn_or_kth(X, y, knn_or_kth):
    results = {choice: {'accuracies': [], 'scores': [], 'labels': []} for choice in knn_or_kth}
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    for choice in knn_or_kth:
        detector = KNNAnomalyDetector(k=10, distance_func=euclidean_distance, knn_or_kth=choice)
        for train_index, test_index in skf.split(X, y):
            X_train, X_test = X[train_index], X[test_index]
            y_train, y_test = y[train_index], y[test_index]

            detector.fit(X_train[y_train == 1])

            predictions = detector.predict(X_test, detector.threshold_)
            scores = detector.decision_function(X_test)
            scores = -scores
            precision, recall, _ = precision_recall_curve(y_test, scores)
            pr_auc = auc(recall, precision)

            results[choice]['accuracies'].append(np.mean(predictions == y_test))
            results[choice]['scores'].extend(scores)
            results[choice]['labels'].extend(y_test)

    approach_labels = ['Kth-Nearest Neighbour' if choice == 0 else 'K-Nearest Neighbours' for choice in knn_or_kth]
    accuracies = [np.mean(results[choice]['accuracies']) for choice in knn_or_kth]
    plt.figure(figsize=(10, 5))
    sns.barplot(x=approach_labels, y=accuracies)
    plt.title('Accuracy by KNN and KTH Approach')
    plt.xlabel('Approach')
    plt.ylabel('Accuracy')
    plt.show()
    
    plt.figure(figsize=(10, 8))
    for choice in results:
        scores = results[choice]['scores']
        labels = results[choice]['labels']
        if choice == 1:
            i = 'K-Nearest Neighbours'
        else:
            i = 'Kth-Nearest Neighbours'
        plot_PR_curve(scores, labels, f'KNN K={i}')
    plt.title('Precision-Recall curves')
    plt.show()
    
    
def train_for_anomalies_and_non_anomalies(X, y, a_or_na):
    results = {choice: {'accuracies': [], 'scores': [], 'labels': []} for choice in a_or_na}
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    detector = KNNAnomalyDetector(k=10, distance_func=euclidean_distance, knn_or_kth=1)
    
    for choice in a_or_na:
        for train_index, test_index in skf.split(X, y):
            X_train, X_test = X[train_index], X[test_index]
            y_train, y_test = y[train_index], y[test_index]

            if choice == 1:
                detector.fit(X_train)
            else:
                detector.fit(X_train[y_train == 1])

            predictions = detector.predict(X_test, detector.threshold_)
            scores = detector.decision_function(X_test)
            scores = -scores
            precision, recall, _ = precision_recall_curve(y_test, scores)
            pr_auc = auc(recall, precision)

            results[choice]['accuracies'].append(np.mean(predictions == y_test))
            results[choice]['scores'].extend(scores)
            results[choice]['labels'].extend(y_test)

    approach_labels = ['Non-anomaly training' if choice == 0 else 'Anomaly training' for choice in a_or_na]
    accuracies = [np.mean(results[choice]['accuracies']) for choice in a_or_na]
    plt.figure(figsize=(10, 5))
    sns.barplot(x=approach_labels, y=accuracies)
    plt.title('Accuracy anomalies and non-anomalies training')
    plt.xlabel('Approach')
    plt.ylabel('Accuracy')
    plt.show()

    plt.figure(figsize=(10, 8))
    for i in results:
        scores = results[i]['scores']
        labels = results[i]['labels']
        if i == 1:
            i = 'Anomalies-including training'
        else:
            i = 'Non-anomalies training'
        plot_PR_curve(scores, labels, f'KNN K={i}')
    plt.title('Precision-Recall curves')
    plt.show()
