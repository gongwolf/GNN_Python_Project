# %% [markdown]
# ### import need packages
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.utils import shuffle

import sklearn as sk
import numpy as np
from sklearn import tree
import time


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
    print('KNN k:{} [all features] Accuracy: {:0.3%}' .format(
        k, accuracy_score(test_data_Y, y_pred)))
    return accuracy_score(test_data_Y, y_pred)


# %% [markdown]
# ### Read Files from csv files
names = [
    'age',
    'workclass',
    'education',
    'capital-gain',
    'capital-loss',
    'hours-per-week',
    'income'
]


train_data = pd.read_csv('data_normal/adult.data_6att_clean_sample_encoded.csv',
                         sep=r"\s*,\s*", names=names, engine='python')

test_data = pd.read_csv('data_normal/adult.test_6att_clean_sample_encoded.csv',
                        sep=r"\s*,\s*", names=names, engine='python')

print(train_data.shape, " ", test_data.shape)

# %% [markdown]
# ### find the max and min value for each colunm in train data and test data
train_max = train_data.max(axis=0)
train_min = train_data.min(axis=0)
train_max_min = pd.concat([train_max, train_min], axis=1)


test_max = test_data.max(axis=0)
test_min = test_data.min(axis=0)
test_max_min = pd.concat([test_max, test_min], axis=1)

train_min.loc["age"], " ", train_max.loc["age"]

# %%
train_data.head(10)

# %% [markdown]
# ### Normalize the data
normal_train_data = train_data.copy()
for feature_name in train_max_min.index.values[:-1]:
    # print(feature_name)
    normal_train_data[feature_name] = (
        train_data[feature_name] - train_min.loc[feature_name]) / train_max.loc[feature_name]

normal_test_data = test_data.copy()
for feature_name in test_max_min.index.values[:-1]:
    # print(feature_name)
    normal_test_data[feature_name] = (
        test_data[feature_name] - test_min.loc[feature_name]) / test_max.loc[feature_name]

normal_train_data.head(10)

# %% [markdown]
# ### save the files
normal_train_data.to_csv(
    'data_normal/adult.data_6att_clean_sample_encoded_01norm.csv', index=False, header=False)

normal_test_data.to_csv(
    'data_normal/adult.test_6att_clean_sample_encoded_01norm.csv', index=False, header=False)


# %% [markdown]
iter = 10
dt_att_acc = [0]*10
rf_att_acc = [0]*10
svc_att_acc = [0]*10
for i in range(iter):
    print(i, " : ")
    dt_att_acc[i] = DTModel(normal_train_data, normal_test_data)
    rf_att_acc[i] = RFModel(normal_train_data, normal_test_data)
    svc_att_acc[i] = SVCModel(normal_train_data, normal_test_data)
# %%
print(dt_att_acc)
print(rf_att_acc)
print(svc_att_acc)
print("{:0.2%} {:0.2%} {:0.2%}".format(sum(dt_att_acc)/10,
                                       sum(rf_att_acc)/10, sum(svc_att_acc)/10))

# %%

# %%
k_nn_att_acc = [0]*10
for k in range(1, 11):
    k_nn_att_acc[k-1] = KNNModel(normal_train_data, normal_test_data, k=k)


# %%
k_nn_att_acc


# %%
knn_result_dict={}
for feature_name in names[:-1]:
    featured_knn_acc = [0]*5
    for k in range(1, 10, 2):
        temp_normal_train_data = normal_train_data[[feature_name, "income"]]
        temp_normal_test_data = normal_test_data[[feature_name, "income"]]
        print(feature_name, ' ', k, " ",int(k/2))
        i = int(k/2)
        featured_knn_acc[i] = KNNModel(temp_normal_train_data,
                                      temp_normal_test_data, k)
    knn_result_dict.update({feature_name:featured_knn_acc})

# %%
knn_result_dict

#%%
