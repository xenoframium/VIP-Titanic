#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Feb  9 09:34:39 2020

@author: tsandhu
"""

# Pandas is a nice utilitiy that enables some easy data manipulation, especially from a csv
import pandas as pd
# Numpy lets us work with arrays
import numpy as np
import re
# Sklearn provides various modules with a common API
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix

train_data = pd.read_csv('train.csv')
test_data = pd.read_csv('test.csv')

train_data.columns[train_data.isna().any()].tolist()

train_data.set_index(keys=['PassengerId'], drop=True, inplace=True)

test_data.set_index(keys=['PassengerId'], drop=True, inplace=True)

train_nan_map = {'Fare': train_data['Fare'].mean(), 'Embarked': train_data['Embarked'].mode()[0]}
test_nan_map = {'Fare': test_data['Fare'].mean(), 'Embarked': test_data['Embarked'].mode()[0]}

train_data.fillna(value=train_nan_map, inplace=True)
test_data.fillna(value=test_nan_map, inplace=True)

#Southampton Cherbourg Queenstown

columns_map = {'Sex': {'male': 0, 'female': 1}}

train_data.replace(columns_map, inplace=True)
test_data.replace(columns_map, inplace=True)

titles = set()

for index, row in train_data.iterrows():
	m = re.search(", ([^\\.]*)", row['Name'])
	title = m.group(1)
	titles.add(title)

for index, row in test_data.iterrows():
	m = re.search(", ([^\\.]*)", row['Name'])
	title = m.group(1)
	titles.add(title)

titles.add("C")
titles.add("Q")
titles.add("S")
new_cols = dict()
for title in titles:
	new_cols[title] = [0 for x in range(0, train_data.shape[0])]


for index, row in train_data.iterrows():
	new_cols[row['Embarked']][index - 1] = 1
	m = re.search(", ([^\\.]*)", row['Name'])
	title = m.group(1)
	new_cols[title][index - 1] = 1

for column in new_cols.keys():
	train_data[column] = new_cols[column]

new_cols = dict()
for title in titles:
	new_cols[title] = [0 for x in range(0, test_data.shape[0])]

for index, row in test_data.iterrows():
	new_cols[row['Embarked']][index - 1 - train_data.shape[0]] = 1
	m = re.search(", ([^\\.]*)", row['Name'])
	title = m.group(1)
	new_cols[title][index - 1 - train_data.shape[0]] = 1

for column in new_cols.keys():
	test_data[column] = new_cols[column]

both = train_data.append(test_data)

title_age_sum = dict()
title_count = dict()

for title in titles:
	title_age_sum[title] = 0
	title_count[title] = 0

for index, row in both.iterrows():
	m = re.search(", ([^\\.]*)", row['Name'])
	title = m.group(1)
	if np.isnan(row['Age']):
		continue
	title_count[title] += 1
	title_age_sum[title] += row['Age']

for index, row in test_data.iterrows():
	m = re.search(", ([^\\.]*)", row['Name'])
	title = m.group(1)
	if not np.isnan(row['Age']):
		continue
	test_data.at[index, 'Age'] = title_age_sum[title] / title_count[title]


for index, row in train_data.iterrows():
	m = re.search(", ([^\\.]*)", row['Name'])
	title = m.group(1)
	if not np.isnan(row['Age']):
		continue
	train_data.at[index, 'Age'] = title_age_sum[title] / title_count[title]

del both['Name']
del both['Cabin']
del both['Ticket']
del both['Embarked']
del both['Survived']

mn = both.min()
mx = both.max()

del test_data['Name']
del test_data['Cabin']
del test_data['Ticket']
del test_data['Embarked']

y_train = train_data.loc[:, 'Survived']

del train_data['Name']
del train_data['Cabin']
del train_data['Ticket']
del train_data['Embarked']
del train_data['Survived']

test_data=(test_data-mn)/(mn+mx)
train_data=(train_data-mn)/(mn+mx)

X_train = train_data.loc[:]

X_train, X_test, y_train, y_test = train_test_split(X_train, y_train, test_size=0.33, random_state=10)

#End of processing

sgd_clf =  SGDClassifier( max_iter=90)
sgd_clf.fit(X_train.values, y_train.values)
print(sgd_clf.score(X_test.values, y_test.values))
y_pred = sgd_clf.predict(X_test.values)
y_truth = y_test.values
FP = len([x for x in range(len(y_pred)) if y_truth[x] == False and y_pred[x] == True])
FN = len([x for x in range(len(y_pred)) if y_truth[x] == True and y_pred[x] == False])
print("FP: " + str(FP / len(y_truth)))
print("FN: " + str(FN / len(y_truth)))


import itertools
import matplotlib.pyplot as plt

def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()


# Compute confusion matrix
cnf_matrix = confusion_matrix(y_truth, y_pred)
class_names=['0', '1']
np.set_printoptions(precision=2)

# Plot non-normalized confusion matrix
plt.figure()
plot_confusion_matrix(cnf_matrix, classes=class_names,
                      title='Confusion matrix, without normalization')

plt.show()

plt.scatter(FP,FN, color='b')
plt.xlabel('false positives')
plt.ylabel('true positives')
plt.show()

predictions = sgd_clf.predict(test_data.values)
type(predictions)




def find_pareto(data):
    is_Pareto = np.ones(data.shape[0], dtype = bool)
    for i, c in enumerate(data):
        # Keep any point with a lower cost
        if is_Pareto[i]:
            # This is where you would change for miniminzation versus maximization 

            # Minimization
            is_Pareto[is_Pareto] = np.any(data[is_Pareto]<c, axis=1)  

            # Maximization
            #is_Pareto[is_Pareto] = np.any(data[is_Pareto]>c, axis=1)  

            # And keep self
            is_Pareto[i] = True  

    # Downsample from boolean array
    Pareto_data = data[is_Pareto, :]

    # Sort data
    Pareto_out =  Pareto_data[np.argsort(Pareto_data[:,0])]

    #return is_Pareto
    return Pareto_out

# Create random list of values
N=20
myData = np.random.random((N,2))

# Include the trival Pareto points, i.e. always on or off
myData = np.vstack(([[0,1],[1,0]], myData))

# Use above routine to find pareto points
myPareto=find_pareto(myData)

# Calculate the Area under the Curve as a Riemann sum
auc = np.sum(np.diff(myPareto[:,0])*myPareto[0:-1,1])

# Create figure
plt.figure()

# Make sure font sizes are large enough to read in the presentation
plt.rcParams.update({'font.size': 14})

# Plot all points
plt.scatter(myData[:,0],myData[:,1],)

# Plot Pareto steps. note 'post' for minimization 'pre' for maximization
plt.step(myPareto[:,0], myPareto[:,1], where='post')
#plt.step(myPareto[:,0], myPareto[:,1], where='pre')

# Make sure you include labels
# Minimization
plt.title('Example of a Minimization Result\n with AUC = ' + str(auc))
plt.xlabel('False Negative')
plt.ylabel('False Positive')

# Maximization
#plt.title('Example of a Maximization Result')
#plt.xlabel('True Negative')
#plt.ylabel('True Positive')

plt.show()

find_pareto(predictions)