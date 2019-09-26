# To add a new cell, type '#%%'
# To add a new markdown cell, type '#%% [markdown]'
# %% [markdown]
'''
# Load the needed libs and dataset <br>
# url of the adult dataset:  https://archive.ics.uci.edu/ml/datasets/Adult
'''

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


# %%
# names of colnum
names = [
    'age',
    'workclass',
    'fnlwgt',
    'education',
    'education-num',
    'marital-status',
    'occupation',
    'relationship',
    'race',
    'sex',
    'capital-gain',
    'capital-loss',
    'hours-per-week',
    'native-country',
    'income',
]
# read the training data and trim the space in each field
train_data = pd.read_csv('data/adult.data', sep="\s*,\s*",
                         names=names, engine='python')
# read the testing data and trim the space in each field
test_data = pd.read_csv('data/adult.test', sep="\s*,\s*",
                        names=names, engine='python')

# in the test dataset, the colnum of the 'income' has a surplus period.
test_data['income'].replace(
    regex=True, inplace=True, to_replace=r'\.', value=r'')
print(train_data.info())

# %%
# %%
print('Missing values in training data')
for i, j in zip(train_data.columns, (train_data.values.astype(str) == '?').sum(axis=0)):
    if j > 0:
        print(str(i) + ': ' + str(j) + ' out of ' +
              str(len(train_data))+' records')


# %%
print('Missing values in test data')
for i, j in zip(test_data.columns, (test_data.values.astype(str) == '?').sum(axis=0)):
    if j > 0:
        print(str(i) + ': ' + str(j) + ' out of ' +
              str(len(test_data))+' records')

# %%
# %% [markdown]
'''
# ## removing the rows have the missing values from the traning dataset
'''
clean_train_data = train_data.copy()
befor_train = len(clean_train_data)
for colnum in names:
    be = len(clean_train_data)
    clean_train_data = clean_train_data[(clean_train_data[colnum] != '?')]
    af = len(clean_train_data)
    if be != af:
        print("{}: before delete: {}     after delete:{}".format(colnum, be, af))
after_train = len(clean_train_data)
print("size of training dataset: before removing: {}    after delete:{}".format(
    befor_train, after_train))


# %% [markdown]
'''
# ## removing the rows have the missing values from the test dataset
'''
clean_test_data = test_data.copy()
befor_test = len(clean_test_data)
for colnum in names:
    be = len(clean_test_data)
    clean_test_data = clean_test_data[(clean_test_data[colnum] != '?')]
    af = len(clean_test_data)
    if be != af:
        print("{}: before delete: {}     after delete:{}".format(colnum, be, af))
after_test = len(clean_test_data)
print("size of test dataset: before removing: {}     after delete:{}".format(
    befor_test, after_test))


# %%
attr_list = ['age', 'workclass', 'education', 'capital-gain',
             'capital-loss', 'hours-per-week', 'income']
train_data_6att_clean = clean_train_data[map(
    lambda x:x in attr_list, list(clean_train_data.columns))].copy()
test_data_6att_clean = clean_test_data[map(
    lambda x:x in attr_list, list(clean_test_data.columns))].copy()

# save the un-encoded filese
train_data_6att_clean.to_csv(
    'data/adult.data_6att_clean.csv', index=False, header=False)
test_data_6att_clean.to_csv(
    'data/adult.test_6att_clean.csv', index=False, header=False)

# %% [markdown]
# ## Setting mapping from the categorical value to nominal value
cat_to_nom_mapping = {}
workclass_mapping = {"Never-worked": 1, "Without-pay": 2,
                     "State-gov": 3, "Local-gov": 4, "Federal-gov": 5,
                     "Self-emp-inc": 6, "Self-emp-not-inc": 7, "Private": 8}

education_mapping = {"Preschool": 1, "1st-4th": 2, "5th-6th": 3,
                     "7th-8th": 4,  "9th": 5, "10th": 6, "11th": 7, "12th": 8,
                     "HS-grad": 9, "Prof-school": 10, "Assoc-acdm": 11,
                     "Assoc-voc": 12, "Some-college": 13, "Bachelors": 14,
                     "Masters": 15, "Doctorate": 16
                     }
income_mapping = {'<=50K': 0, '>50K': 1}
cat_to_nom_mapping.update({"workclass": workclass_mapping})
cat_to_nom_mapping.update({"education": education_mapping})
cat_to_nom_mapping.update({"income": income_mapping})


# %% [markdown]
# mapping and save the files
train_data_6att_clean_encoded = train_data_6att_clean.copy()
test_data_6att_clean_encoded = test_data_6att_clean.copy()

train_data_6att_clean_encoded.replace(cat_to_nom_mapping, inplace=True)
test_data_6att_clean_encoded.replace(cat_to_nom_mapping, inplace=True)

train_data_6att_clean_encoded.to_csv(
    'data/adult.data_6att_clean_encoded.csv', index=False, header=False)
test_data_6att_clean_encoded.to_csv(
    'data/adult.test_6att_clean_encoded.csv', index=False, header=False)

# %% [markdown]
# ## Random sample 1000 rows from train_data_6att_clean_encoded and test_data_6att_clean_encoded
# ## & save the file


def getSamples(train_data, test_data, num_sample=1000):
    seed = int(round(time.time()))
    sample_train = train_data.sample(num_sample*10, random_state=seed)
    sample_test = test_data.sample(num_sample, random_state=seed)
    return sample_train, sample_test


def saveSamples(train_data, test_data):
    train_data.to_csv(
        'data/adult.data_6att_clean_sample_encoded.csv', index=False, header=False)
    test_data.to_csv(
        'data/adult.test_6att_clean_sample_encoded.csv', index=False, header=False)


# %% [markdown]
# ## the function generated the balanced sample data sets
def getBalanceSamples(train_data, test_data, num_sample=1000):
    seed = int(round(time.time()))
    sampled_train_data_one = train_data.loc[train_data['income'] == 1].sample(
        num_sample*5, random_state=seed)
    sampled_train_data_zero = train_data.loc[train_data['income'] == 0].sample(
        num_sample*5, random_state=seed)
    sample_train = shuffle(
        sampled_train_data_one.append(sampled_train_data_zero))
    # sample_train = train_data.sample(num_sample*10, random_state=seed)

    sampled_test_data_one = test_data.loc[test_data['income'] == 1].sample(
        int(num_sample/2), random_state=seed)
    sampled_test_data_zero = test_data.loc[test_data['income'] == 0].sample(
        int(num_sample/2), random_state=seed)
    sample_test = shuffle(sampled_test_data_one.append(sampled_test_data_zero))
    return sample_train, sample_test


train_data_6att_clean_sample_encoded, test_data_6att_clean_sample_encoded = getBalanceSamples(
    train_data_6att_clean_encoded, test_data_6att_clean_encoded)
saveSamples(train_data_6att_clean_sample_encoded,
            test_data_6att_clean_sample_encoded)

assert sum(train_data_6att_clean_sample_encoded['income']
           == 1) == 5000, "wrong sampled training file (1)"
assert sum(train_data_6att_clean_sample_encoded['income']
           == 0) == 5000, "wrong sampled training file (0)"
assert sum(test_data_6att_clean_sample_encoded['income']
           == 1) == 500, "wrong sampled test file (1)"
assert sum(test_data_6att_clean_sample_encoded['income']
           == 0) == 500, "wrong sampled test file (0)"
# %%
dt_att_acc = [0]*10
dt_att_sample_acc = [0]*10
rf_att_acc = [0]*10
rf_att_sample_acc = [0]*10
svc_att_acc = [0]*10
svc_att_sample_acc = [0]*10

for i in range(10):
    print(i, " : ")
    # train_data_6att_clean_sample_encoded, test_data_6att_clean_sample_encoded = getSamples(
    #     train_data_6att_clean_encoded, test_data_6att_clean_encoded)
    train_data_6att_clean_sample_encoded, test_data_6att_clean_sample_encoded = getBalanceSamples(
        train_data_6att_clean_encoded, test_data_6att_clean_encoded)
    saveSamples(train_data_6att_clean_sample_encoded,
                test_data_6att_clean_sample_encoded)

    acurray_6att_dt = DTModel(
        train_data_6att_clean_encoded, test_data_6att_clean_encoded)
    acurray_6att_sample_dt = DTModel(
        train_data_6att_clean_sample_encoded, test_data_6att_clean_sample_encoded)
    acurray_6att_rf = RFModel(
        train_data_6att_clean_encoded, test_data_6att_clean_encoded)
    acurray_6att_sample_rf = RFModel(
        train_data_6att_clean_sample_encoded, test_data_6att_clean_sample_encoded)
    acurray_6att_SVC = SVCModel(
        train_data_6att_clean_encoded, test_data_6att_clean_encoded)
    acurray_6att_sample_SVC = SVCModel(
        train_data_6att_clean_sample_encoded, test_data_6att_clean_sample_encoded)

    # get the results
    dt_att_acc[i] = acurray_6att_dt
    dt_att_sample_acc[i] = acurray_6att_sample_dt
    rf_att_acc[i] = acurray_6att_rf
    rf_att_sample_acc[i] = acurray_6att_sample_rf
    svc_att_acc[i] = acurray_6att_SVC
    svc_att_sample_acc[i] = acurray_6att_sample_SVC

# %% [markdown]
# ## print the reable results
print(dt_att_acc)
print(dt_att_sample_acc)
print(rf_att_acc)
print(rf_att_sample_acc)
print(svc_att_acc)
print(svc_att_sample_acc)

print("{} {} {} {} {} {}".format(sum(dt_att_acc)/10, sum(dt_att_sample_acc)/10,
                                 sum(rf_att_acc)/10, sum(rf_att_sample_acc)/10,
                                 sum(svc_att_acc)/10, sum(svc_att_sample_acc)/10))


# %%
k_nn_att_acc = [0]*10
k_nn_att_sample_acc = [0]*10
for i in range(10):
    print(i, " : ")
    train_data_6att_clean_sample_encoded, test_data_6att_clean_sample_encoded = getBalanceSamples(
        train_data_6att_clean_encoded, test_data_6att_clean_encoded)
    saveSamples(train_data_6att_clean_sample_encoded,
                test_data_6att_clean_sample_encoded)

    k_nn_att_acc

    for k in range(1, 11):
        acurray_6att_knn = KNNModel(
            train_data_6att_clean_encoded, test_data_6att_clean_encoded, k)
        k_nn_att_acc[k-1] = k_nn_att_acc[k-1] + acurray_6att_knn
    for k in range(1, 11):
        acurray_6att_sample_knn = KNNModel(
            train_data_6att_clean_sample_encoded, test_data_6att_clean_sample_encoded, k)
        k_nn_att_sample_acc[k-1] = k_nn_att_sample_acc[k -
                                                       1] + acurray_6att_sample_knn

print(k_nn_att_acc)
print(k_nn_att_sample_acc)
# %%
print(np.asarray(k_nn_att_acc)/10)
print(np.asarray(k_nn_att_sample_acc)/10)

# %%
