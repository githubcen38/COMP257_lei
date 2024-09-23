# -*- coding: utf-8 -*-
"""
Created on Sun Sep 22 14:04:18 2024

@author: leiqa
"""

#import the necessary libraries
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import numpy as np
from sklearn.datasets import fetch_openml
#import warnings
#warnings.filterwarnings(action='ignore', category=FutureWarning)

#Retrieve and load the mnist_784 dataset of 70,000 instances.
mnist = fetch_openml('mnist_784', version=1, as_frame=False)
print(mnist.data.shape)
X = mnist['data']
y = mnist['target']

#Display each digit. 

def display_each_digit(X, y, n=10):
    plt.figure(figsize=(10, 1))
    for i in range(n):
        plt.subplot(1, n, i + 1)
        plt.imshow(X[i].reshape(28, 28), cmap='binary')
        plt.axis('off')
        plt.title(f'{y[i]}')
    plt.show()

display_each_digit(X, y)

#Use PCA to retrieve the 1th and 2th principal component and output their explained variance ratio
n_components = 2
pca = PCA(n_components=n_components)
X_pca = pca.fit(X)
explained_variance_ratio = pca.explained_variance_ratio_
print(f'Explained_variance_ratio: {explained_variance_ratio}')

#Plot the projections of the  1th and 2th principal component onto a 1D hyperplane.
X_pca = pca.transform(X)
for i in range(10):
    plt.plot(X_pca[y == str(i), 0], np.zeros_like(X_pca[y == str(i), 0]), label=str(i))
plt.legend()
plt.show()
        
#Use Incremental PCA to reduce the dimensionality of the MNIST dataset down to 154 dimensions.
n_components_154 =  154
pca_5 = PCA(n_components=n_components_154)
X_pca_5 = pca_5.fit_transform(X)
X_rebuilt = pca_5.inverse_transform(X_pca_5)

#Display the original and compressed digits from (5).

def display_original_compressed(X, X_rebuilt, n=10):
    plt.figure(figsize=(10, 2))
    for i in range(n):
        plt.subplot(2, n, i + 1)
        plt.imshow(X[i].reshape(28, 28), cmap='binary')
        plt.axis('off')
        
        plt.subplot(2, n, n + i + 1)
        plt.imshow(X_rebuilt[i].reshape(28, 28), cmap='binary')
        plt.axis('off')
    plt.show()

display_original_compressed(X, X_rebuilt)



