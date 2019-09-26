
#%% 
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.utils import shuffle

import sklearn as sk
import numpy as np
import pandas as pd
from sklearn import tree
import time
# %% [markdown]
# ### Define the Decision Tree Model


def DTModel(train_data, test_data):
    # Split the data frame
    train_data_X = train_data.iloc[:, :-1]
    train_data_Y = train_data.iloc[:, -1:]

    test_data_X = test_data.iloc[:, :-1]
    test_data_Y = test_data.iloc[:, -1:]
    # print(train_data_X.shape, train_data_Y.shape,
    #       test_data_X.shape, test_data_Y.shape)

    # Train and predict the model
    clf = tree.DecisionTreeClassifier()
    clf.fit(train_data_X, train_data_Y)
    y_pred = clf.predict(test_data_X)
    print('DT MODEL  [all features] Accuracy: {:0.2%}' .format(
        accuracy_score(test_data_Y, y_pred)))
    return accuracy_score(test_data_Y, y_pred)

# %% [markdown]
# ### Random Forest Model


def RFModel(train_data, test_data):
    # Split the data frame
    train_data_X = train_data.iloc[:, :-1]
    train_data_Y = train_data.iloc[:, -1:]

    test_data_X = test_data.iloc[:, :-1]
    test_data_Y = test_data.iloc[:, -1:]
    # print(train_data_X.shape, train_data_Y.shape,
    #       test_data_X.shape, test_data_Y.shape)

    # Train and predict the model
    clf = RandomForestClassifier(n_estimators=100)
    clf.fit(train_data_X, train_data_Y.values.ravel())
    y_pred = clf.predict(test_data_X)
    print('RF MODEL  [all features] Accuracy: {:0.2%}' .format(
        accuracy_score(test_data_Y, y_pred)))
    return accuracy_score(test_data_Y, y_pred)

# %% [markdown]
# ### SVC


def SVCModel(train_data, test_data):
    # Split the data frame
    train_data_X = train_data.iloc[:, :-1]
    train_data_Y = train_data.iloc[:, -1:]

    test_data_X = test_data.iloc[:, :-1]
    test_data_Y = test_data.iloc[:, -1:]
    # print(train_data_X.shape, train_data_Y.shape,
    #       test_data_X.shape, test_data_Y.shape)

    # Train and predict the model
    clf = SVC(gamma='scale')
    clf.fit(train_data_X, train_data_Y.values.ravel())
    y_pred = clf.predict(test_data_X)
    print('SVC MODEL  [all features] Accuracy: {:0.2%}' .format(
        accuracy_score(test_data_Y, y_pred)))
    return accuracy_score(test_data_Y, y_pred)

# %% [markdown]
# ### KNN


def KNNModel(train_data, test_data, k=5):
    # Split the data frame
    train_data_X = train_data.iloc[:, :-1]
    train_data_Y = train_data.iloc[:, -1:]

    test_data_X = test_data.iloc[:, :-1]
    test_data_Y = test_data.iloc[:, -1:]

    neigh = KNeighborsClassifier(n_neighbors=k)
    neigh.fit(train_data_X, train_data_Y.values.ravel())
    y_pred = neigh.predict(test_data_X)
    print('KNN k:{} [all features] Accuracy: {:0.2%}' .format(
        k, accuracy_score(test_data_Y, y_pred)))
    return accuracy_score(test_data_Y, y_pred)