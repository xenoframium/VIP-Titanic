# Pandas is a nice utilitiy that enables some easy data manipulation, especially from a csv
import pandas as pd
# Numpy lets us work with arrays
import numpy as np
import re
# Sklearn provides various modules with a common API
from sklearn import svm, tree, neighbors, neural_network
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

from sklearn.linear_model import PassiveAggressiveClassifier
from sklearn.datasets import make_classification

PAC_clf = PassiveAggressiveClassifier(max_iter=1000, random_state=0, tol=1e-4)
PAC_clf.fit(X_train.values, y_train.values)
print(PAC_clf.score(X_test.values, y_test.values))
y_pred = PAC_clf.predict(X_test.values)
y_truth = y_test.values
FP = len([x for x in range(len(y_pred)) if y_truth[x] == False and y_pred[x] == True])
FN = len([x for x in range(len(y_pred)) if y_truth[x] == True and y_pred[x] == False])
print("FP: " + str(FP / len(y_truth)))
print("FN: " + str(FN / len(y_truth)))