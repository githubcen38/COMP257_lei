# -*- coding: utf-8 -*-
"""
Created on Sun Sep 22 16:35:06 2024

@author: leiqa
"""

#import the necessary libraries
from sklearn.datasets import make_swiss_roll
import matplotlib.pyplot as plt
from sklearn.decomposition import KernelPCA
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
import numpy as np


#Generate Swiss roll dataset.
X, y = make_swiss_roll(n_samples=1000, noise=0.2, random_state=42)

#Plot the resulting generated Swiss roll dataset.
figure = plt.figure(figsize=(6, 6))
axis = figure.add_subplot(111, projection='3d')
axis.scatter(X[:, 0], X[:, 1], X[:, 2], c=y, cmap=plt.cm.Spectral)
plt.show()

#Use Kernel PCA (kPCA) with linear kernel (2 points), a RBF kernel (2 points), 
#and a sigmoid kernel (2 points).

linear_kernel = KernelPCA(n_components=2, kernel='linear')
X_linear = linear_kernel.fit_transform(X)

rbf_kernel = KernelPCA(n_components=2, kernel='rbf', gamma=0.04)
X_rbf = rbf_kernel.fit_transform(X)

sigmoid_Kernel = KernelPCA(n_components=2, kernel='sigmoid', gamma=0.001, coef0=1)
X_sigmoid = sigmoid_Kernel.fit_transform(X)

#Plot the kPCA results of applying the linear kernel (2 points), a RBF kernel (2 points), 
# and a sigmoid kernel (2 points) from (3). Explain and compare the results [6 points]
plt.figure(figsize=(15, 5))

plt.subplot(131)
plt.scatter(X_linear[:, 0], X_linear[:, 1], c=y, cmap=plt.cm.Spectral)
plt.title('Linear')

plt.subplot(132)
plt.scatter(X_rbf[:, 0], X_rbf[:, 1], c=y, cmap=plt.cm.Spectral)
plt.title('RBF kernel')

plt.subplot(133)
plt.scatter(X_sigmoid[:, 0], X_sigmoid[:, 1], c=y, cmap=plt.cm.Spectral)
plt.title('Sigmoid')

plt.show()

#Using kPCA and a kernel of your choice, apply Logistic Regression for classification. 
clf = Pipeline([
    ("kpca", KernelPCA(n_components=2)),
    ("log__reg", LogisticRegression())
    ])

param_grid = [{
  "kpca__gamma": np.linspace(0.03, 0.05, 10),
  "kpca__kernel": ['rbf', 'sigmoid', 'linear']
}]

grid_search = GridSearchCV(clf, param_grid, cv=3)
grid_search.fit(X,y)

print(grid_search.best_params_)


