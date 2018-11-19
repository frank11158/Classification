import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc

dataset = pd.read_csv('./Data/data.csv')

# Data preprocessing
dataset['outlook'] = LabelEncoder().fit_transform(dataset['outlook'])
dataset['hangingOut'] = LabelEncoder().fit_transform(dataset['hangingOut'])

X = dataset.drop('hangingOut', axis = 1)
y = dataset['hangingOut']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 0)

sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

# Decision-tree classifier
from sklearn.tree import DecisionTreeClassifier
DTC = DecisionTreeClassifier(random_state = 0)
DTC.fit(X_train, y_train)
y_pred = DTC.predict(X_test)
print("Decision Tree:")
print(classification_report(y_test, y_pred))
print(confusion_matrix(y_test, y_pred))
fpr, tpr, thresholds = roc_curve(y_test, y_pred)
print(auc(fpr, tpr))

# Kernel SVC
from sklearn.svm import SVC
SVC = SVC(kernel = 'rbf', random_state = 0)
SVC.fit(X_train, y_train)
y_pred = SVC.predict(X_test)
print("Kernel SVC:")
print(classification_report(y_test, y_pred))
print(confusion_matrix(y_test, y_pred))
fpr, tpr, thresholds = roc_curve(y_test, y_pred)
print(auc(fpr, tpr))

# Random-forest classifier
from sklearn.ensemble import RandomForestClassifier
RFC = RandomForestClassifier(n_estimators = 200, criterion = 'entropy', random_state = 0)
RFC.fit(X_train, y_train)
y_pred = RFC.predict(X_test)
print("Random Forest:")
print(classification_report(y_test, y_pred))
print(confusion_matrix(y_test, y_pred))
fpr, tpr, thresholds = roc_curve(y_test, y_pred)
print(auc(fpr, tpr))

# Naive-bayes
from sklearn.naive_bayes import GaussianNB
NB = GaussianNB()
NB.fit(X_train, y_train)
y_pred = NB.predict(X_test)
print("Naive Bayes:")
print(classification_report(y_test, y_pred))
print(confusion_matrix(y_test, y_pred))
fpr, tpr, thresholds = roc_curve(y_test, y_pred)
print(auc(fpr, tpr))

# Decision tree relationship graph
from sklearn.externals.six import StringIO  
from IPython.display import Image  
from sklearn.tree import export_graphviz
import pydotplus
dot_data = StringIO()
export_graphviz(DTC, out_file=dot_data,
                class_names=['Yes','No'],
                feature_names=['outlook','humidity','temperature','time'],
                filled=True, rounded=True,
                special_characters=True)
graph = pydotplus.graph_from_dot_data(dot_data.getvalue())  
Image(graph.create_png())
graph.write_png("graph1.png")