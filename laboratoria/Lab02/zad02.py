from sklearn import datasets
from sklearn.decomposition import PCA
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt


iris = datasets.load_iris()
X = pd.DataFrame(iris.data, columns=iris.feature_names)

pca_full = PCA().fit(X)

explained_variance = pca_full.explained_variance_ratio_
print("Explained variance ratios:", explained_variance)

cumulative_variance = explained_variance.cumsum()
print("Cumulative explained variance:", cumulative_variance)

num_components = np.argmax(cumulative_variance >= 0.95) + 1
print("Number of components to retain at least 95% of the variance:", num_components)



n_components_plot = num_components

pca_plot = PCA(n_components=n_components_plot).fit(X)
X_pca = pca_plot.transform(X)


if n_components_plot == 2:
    plt.figure(figsize=(8, 6))
    scatter = plt.scatter(X_pca[:, 0], X_pca[:, 1], c=iris.target, cmap='viridis', edgecolor='k')
    plt.xlabel('PC1')
    plt.ylabel('PC2')
    plt.title('PCA of Iris Dataset (2 Components)')
    plt.colorbar(scatter, label='Iris Species')
    plt.show()
elif n_components_plot == 3:
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection='3d')
    scatter = ax.scatter(X_pca[:, 0], X_pca[:, 1], X_pca[:, 2], c=iris.target, cmap='viridis', edgecolor='k')
    ax.set_xlabel('PC1')
    ax.set_ylabel('PC2')
    ax.set_zlabel('PC3')
    plt.title('PCA of Iris Dataset (3 Components)')
    plt.colorbar(scatter, label='Iris Species')
    plt.show()
