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
import sklearn as sk
import numpy as np
import pandas as pd
from sklearn import tree

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


# %% [markdown]
'''
# ## print the labels of the colnum "workclass"
'''
print(set(train_data['workclass']))


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


# %% [markdown]
# save the files without encoding
# clean_train_data.to_csv('data/adult.data_clearn', index=False, header=False)
# clean_test_data.to_csv('data/adult.test_clean', index=False, header=False)

# clean_train_data_except_X_saving = clean_train_data[map(
#     lambda x:x not in ['race', 'sex'], list(clean_train_data.columns))].copy()
# clean_test_data_except_X_saving = clean_test_data[map(
#     lambda x:x not in ['race', 'sex'], list(clean_test_data.columns))].copy()
# clean_train_data_except_X_saving.to_csv(
#     'data/adult.data_clean_nosexrace', index=False, header=False)
# clean_test_data_except_X_saving.to_csv(
#     'data/adult.test_clean_nosexrace', index=False, header=False)

# %% [markdown]
'''
# ## convert the categorical value to int
# build the mapping from categorical string to nominal value <br>
# call the df.replace function to Encoding Categorical Values (https://pbpython.com/categorical-encoding.html)
'''
cat_to_nom_mapping = {}
for col in clean_test_data.columns:
    if clean_train_data[col].dtype == object:
        # set of labels of the categorical columns
        labels = set(clean_train_data[col])
        mapping_labels = {key: value for value, key in enumerate(labels)}
        print(col, mapping_labels)
        cat_to_nom_mapping.update({col: mapping_labels})

print(cat_to_nom_mapping)
clean_train_data.replace(cat_to_nom_mapping, inplace=True)
clean_test_data.replace(cat_to_nom_mapping, inplace=True)
print(clean_train_data.info())
print(clean_test_data.info())


# %% [markdown]
# # save the files with encoding
# clean_train_data.to_csv('data/adult.data_encoded_clearn',
#                         index=False, header=False)
# clean_test_data.to_csv('data/adult.test_encoded_clean',
#                        index=False, header=False)

# clean_train_data_except_X_saving = clean_train_data[map(
#     lambda x:x not in ['race', 'sex'], list(clean_train_data.columns))].copy()
# clean_test_data_except_X_saving = clean_test_data[map(
#     lambda x:x not in ['race', 'sex'], list(clean_test_data.columns))].copy()
# clean_train_data_except_X_saving.to_csv(
#     'data/adult.data_encoded_clean_nosexrace', index=False, header=False)
# clean_test_data_except_X_saving.to_csv(
#     'data/adult.test_encoded_clean_nosexrace', index=False, header=False)

# %% [markdown]
'''
# ### Decision Tree Model
'''

# all features
# Split the data frame
clean_train_data_X = clean_train_data.iloc[:, :-1]
clean_train_data_Y = clean_train_data.iloc[:, -1:]

clean_test_data_X = clean_test_data.iloc[:, :-1]
clean_test_data_Y = clean_test_data.iloc[:, -1:]
print(clean_train_data_X.shape, clean_train_data_Y.shape,
      clean_test_data_X.shape, clean_test_data_Y.shape)


# Train and predict the model
clf = tree.DecisionTreeClassifier()
clf.fit(clean_train_data_X, clean_train_data_Y)
y_pred = clf.predict(clean_test_data_X)
print('[all features] Accuracy: {:0.2%}' .format(
    accuracy_score(clean_test_data_Y, y_pred)))

# single features
for col in clean_train_data_X.columns:
    clean_train_data_single_X = clean_train_data_X[col].values.reshape(-1, 1)
    clean_test_data_single_X = clean_test_data[col].values.reshape(-1, 1)
    clf = tree.DecisionTreeClassifier()
    clf.fit(clean_train_data_single_X, clean_train_data_Y)
    y_pred = clf.predict(clean_test_data_single_X)
    print('[{}] Accuracy: {:0.2%}' .format(
        col, accuracy_score(clean_test_data_Y, y_pred)))

# except race and sex
clean_train_data_except_X = clean_train_data_X[map(
    lambda x:x not in ['race', 'sex'], list(clean_train_data_X.columns))].copy()
clean_test_data_except_X = clean_test_data_X[map(
    lambda x:x not in ['race', 'sex'], list(clean_test_data_X.columns))].copy()
print(clean_train_data_except_X.shape, clean_train_data_Y.shape,
      clean_test_data_except_X.shape, clean_test_data_Y.shape)
clf = tree.DecisionTreeClassifier()
clf.fit(clean_train_data_except_X, clean_train_data_Y)
y_pred = clf.predict(clean_test_data_except_X)
print('[except race and sex] Accuracy: {:0.2%}' .format(
    accuracy_score(clean_test_data_Y, y_pred)))


# %%
# %% [markdown]
'''
# ### Random Forest Model
'''

# all features
# Split the data frame
clean_train_data_X = clean_train_data.iloc[:, :-1]
clean_train_data_Y = clean_train_data.iloc[:, -1:]

clean_test_data_X = clean_test_data.iloc[:, :-1]
clean_test_data_Y = clean_test_data.iloc[:, -1:]
print(clean_train_data_X.shape, clean_train_data_Y.shape,
      clean_test_data_X.shape, clean_test_data_Y.shape)


# Train and predict the model
# Add the .values.ravel() to Y to avoid the warning
clf = RandomForestClassifier(n_estimators=100)
clf.fit(clean_train_data_X, clean_train_data_Y.values.ravel())
y_pred = clf.predict(clean_test_data_X)
print('[all features] Accuracy: {:0.2%}' .format(
    accuracy_score(clean_test_data_Y, y_pred)))

# single features
for col in clean_train_data_X.columns:
    clean_train_data_single_X = clean_train_data_X[col].values.reshape(-1, 1)
    clean_test_data_single_X = clean_test_data[col].values.reshape(-1, 1)
    clf = RandomForestClassifier(n_estimators=100)
    clf.fit(clean_train_data_single_X, clean_train_data_Y.values.ravel())
    y_pred = clf.predict(clean_test_data_single_X)
    print('[{}] Accuracy: {:0.2%}' .format(
        col, accuracy_score(clean_test_data_Y, y_pred)))

# except race and sex
clean_train_data_except_X = clean_train_data_X[map(
    lambda x:x not in ['race', 'sex'], list(clean_train_data_X.columns))].copy()
clean_test_data_except_X = clean_test_data_X[map(
    lambda x:x not in ['race', 'sex'], list(clean_test_data_X.columns))].copy()
print(clean_train_data_except_X.shape, clean_train_data_Y.shape,
      clean_test_data_except_X.shape, clean_test_data_Y.shape)
clf = RandomForestClassifier(n_estimators=100)
clf.fit(clean_train_data_except_X, clean_train_data_Y.values.ravel())
y_pred = clf.predict(clean_test_data_except_X)
print('[except race and sex] Accuracy: {:0.2%}' .format(
    accuracy_score(clean_test_data_Y, y_pred)))


# %%
# %% [markdown]
'''
# ### SVM
'''


# all features
# Split the data frame
clean_train_data_X = clean_train_data.iloc[:, :-1]
clean_train_data_Y = clean_train_data.iloc[:, -1:]

clean_test_data_X = clean_test_data.iloc[:, :-1]
clean_test_data_Y = clean_test_data.iloc[:, -1:]
print(clean_train_data_X.shape, clean_train_data_Y.shape,
      clean_test_data_X.shape, clean_test_data_Y.shape)


# Train and predict the model
# Add the .values.ravel() to Y to avoid the warning
clf = SVC(gamma='scale')
clf.fit(clean_train_data_X, clean_train_data_Y.values.ravel())
y_pred = clf.predict(clean_test_data_X)
print('[all features] Accuracy: {:0.2%}' .format(
    accuracy_score(clean_test_data_Y, y_pred)))

# single features
for col in clean_train_data_X.columns:
    clean_train_data_single_X = clean_train_data_X[col].values.reshape(-1, 1)
    clean_test_data_single_X = clean_test_data[col].values.reshape(-1, 1)
    clf = SVC(gamma='scale')
    clf.fit(clean_train_data_single_X, clean_train_data_Y.values.ravel())
    y_pred = clf.predict(clean_test_data_single_X)
    print('[{}] Accuracy: {:0.2%}' .format(
        col, accuracy_score(clean_test_data_Y, y_pred)))

# except race and sex
clean_train_data_except_X = clean_train_data_X[map(
    lambda x:x not in ['race', 'sex'], list(clean_train_data_X.columns))].copy()
clean_test_data_except_X = clean_test_data_X[map(
    lambda x:x not in ['race', 'sex'], list(clean_test_data_X.columns))].copy()
print(clean_train_data_except_X.shape, clean_train_data_Y.shape,
      clean_test_data_except_X.shape, clean_test_data_Y.shape)
clf = SVC(gamma='scale')
clf.fit(clean_train_data_except_X, clean_train_data_Y.values.ravel())
y_pred = clf.predict(clean_test_data_except_X)
print('[except race and sex] Accuracy: {:0.2%}' .format(
    accuracy_score(clean_test_data_Y, y_pred)))

# %% [markdown]
'''
# ### KNN
'''

# all features
# Split the data frame
clean_train_data_X = clean_train_data.iloc[:, :-1]
clean_train_data_Y = clean_train_data.iloc[:, -1:]

clean_test_data_X = clean_test_data.iloc[:, :-1]
clean_test_data_Y = clean_test_data.iloc[:, -1:]
print(clean_train_data_X.shape, clean_train_data_Y.shape,
      clean_test_data_X.shape, clean_test_data_Y.shape)


# %%
# Train and predict the model
# Add the .values.ravel() to Y to avoid the warning
for k in range(1, 11):
    neigh = KNeighborsClassifier(n_neighbors=k)
    neigh.fit(clean_train_data_X, clean_train_data_Y.values.ravel())
    y_pred = neigh.predict(clean_test_data_X)
    print('[all features] k={} Accuracy: {:0.2%}' .format(
        k, accuracy_score(clean_test_data_Y, y_pred)))
# %%
# single features
for col in clean_train_data_X.columns:
    clean_train_data_single_X = clean_train_data_X[col].values.reshape(-1, 1)
    clean_test_data_single_X = clean_test_data[col].values.reshape(-1, 1)
    neigh = KNeighborsClassifier()
    neigh.fit(clean_train_data_single_X, clean_train_data_Y.values.ravel())
    y_pred = neigh.predict(clean_test_data_single_X)
    print('[{}] Accuracy: {:0.2%}' .format(
        col, accuracy_score(clean_test_data_Y, y_pred)))
# %%
# except race and sex
clean_train_data_except_X = clean_train_data_X[map(
    lambda x:x not in ['race', 'sex'], list(clean_train_data_X.columns))].copy()
clean_test_data_except_X = clean_test_data_X[map(
    lambda x:x not in ['race', 'sex'], list(clean_test_data_X.columns))].copy()
print(clean_train_data_except_X.shape, clean_train_data_Y.shape,
      clean_test_data_except_X.shape, clean_test_data_Y.shape)
for k in range(1, 11):
    neigh = KNeighborsClassifier(n_neighbors=k)
    neigh.fit(clean_train_data_except_X, clean_train_data_Y.values.ravel())
    y_pred = neigh.predict(clean_test_data_except_X)
    print('[except race and sex] k={} Accuracy: {:0.2%}' .format(
        k, accuracy_score(clean_test_data_Y, y_pred)))
# %%
