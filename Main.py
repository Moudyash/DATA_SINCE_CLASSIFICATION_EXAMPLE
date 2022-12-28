
import pandas as pd
import numpy as np

dataset = pd.read_csv("E:\programming\python\DATA_SINCE_CLASSIFICATION_EXAMPLE\drug200.csv")
dataset.head()

dataset['Sex'].replace({
    'M':0,
    'F':1
    
},inplace=True)
dataset['BP'].replace({
    'LOW':0,
    'NORMAL':1,
    'HIGH':2
},inplace=True)
dataset['Cholesterol'].replace({
    'LOW':0,
    'NORMAL':1,
    'HIGH':2
},inplace=True)
dataset['Drug'].replace({
    'drugA':0,
    'drugB':2, 
     'drugC':2,
    'DrugY':3,
     'drugX':4,
},inplace=True)
X = dataset.drop('Drug', axis=1)
y = dataset['Drug']
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30)
from sklearn.neighbors import KNeighborsClassifier


classifier = KNeighborsClassifier(n_neighbors=24)
classifier.fit(X_train, y_train)
y_pred = classifier.predict(X_test)
from sklearn.metrics import classification_report, confusion_matrix ,accuracy_score
print(accuracy_score(y_test, y_pred))
print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))

for i in range(1, 50):
    knn = KNeighborsClassifier(n_neighbors=i)
    knn.fit(X_train, y_train)
    pred_i = knn.predict(X_test)
    print(i," " ,accuracy_score(y_test,pred_i))

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20)
from sklearn.tree import DecisionTreeClassifier
classifier = DecisionTreeClassifier()
classifier.fit(X_train, y_train)
y_pred = classifier.predict(X_test)
from sklearn.metrics import classification_report, confusion_matrix ,accuracy_score
print(accuracy_score(y_test, y_pred))

print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30)
from sklearn.naive_bayes import GaussianNB
GNB = GaussianNB().fit(X_train, y_train)

y_pred = GNB.predict(X_test)
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
print(accuracy_score(y_test, y_pred))

print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))

