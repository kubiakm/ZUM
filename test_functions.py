import scipy
import os
import time
import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import seaborn as sns
import sklearn.metrics

def plot_corr(data):
    #Feature corelation plot  


    sns.set_theme(style="white")
    corr = data.corr()
    mask = np.triu(np.ones_like(corr, dtype=bool))
    f, ax = plt.subplots(figsize=(11, 9))
    cmap = sns.diverging_palette(230, 20, as_cmap=True)
    sns.heatmap(corr, mask=mask, cmap=cmap, vmax=.3, center=0, 
                square=True, linewidths=.5, cbar_kws={"shrink": .5})
    plt.show()

def plot_feature_dist(data):
    # feature distribution plot
    fig, ax = plt.subplots()  

    sns.boxplot(data=data, ax=ax)
    plt.title('Boxplot of Six Features')
    plt.xlabel('Features')
    plt.ylabel('Values')
    plt.show()

def plot_pca_results(data_x, data_y, title):
    pca = PCA(n_components=2)
    pca_data = pca.fit_transform(data_x)

    colors = ['blue' if x == 1 else 'red' for x in data_y]

    plt.figure(figsize=(8, 6))
    plt.scatter(pca_data[:, 0], pca_data[:, 1], c=colors, alpha=0.5)
    plt.title(title)
    plt.xlabel('Principal Component 1')
    plt.ylabel('Principal Component 2')
    plt.grid(True)

    plt.show()

def plot_pca_3d(data_x, data_y, title='3d PCA dataset visualization'):
    # Perform PCA to reduce to 3 dimensions
    pca = PCA(n_components=3)
    X_pca = pca.fit_transform(data_x)

    # Create a DataFrame for better handling
    df = pd.DataFrame(data=X_pca, columns=['PC1', 'PC2', 'PC3'])
    df['Target'] = np.array(data_y)

    # Plotting
    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(111, projection='3d')
    ax.set_title(title)
    scatter = ax.scatter(df['PC1'], df['PC2'], df['PC3'], c=df['Target'], cmap='viridis')

    # Adding labels
    ax.set_xlabel('Principal Component 1')
    ax.set_ylabel('Principal Component 2')
    ax.set_zlabel('Principal Component 3')

    # Adding a legend
    legend_labels = ['Anomalie', 'Prawidłowe']
    legend1 = ax.legend(handles=scatter.legend_elements()[0], labels=legend_labels ,title="Classes")
    ax.add_artist(legend1)

    plt.show()

def plot_PR_curve(scores, labels, alg_name, title='Precision-Recall Curve', single=False):
    precision, recall, _ = sklearn.metrics.precision_recall_curve(labels, scores)
    pr_auc = sklearn.metrics.auc(recall, precision)

    if single:
        plt.figure()

    plt.plot(recall, precision, label=f'{alg_name} (AUC = {pr_auc:.3f})')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve')
    plt.legend()
    #plt.ylim([0.5,1])
    if single:
        plt.show()