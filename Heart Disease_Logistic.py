# -*- coding: utf-8 -*-
"""
Created on Mon Mar 20 17:42:09 2023

@author: hp
"""

import os 
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

os.chdir('C:/Users/hp/Desktop/Codes')
data = pd.read_csv('heart_disease_data.csv')

print(data.head())
print(data.isnull().sum())
print(data.shape)

print(data.dtypes)
corr_ht=data.corr()
print(corr_ht)
sns.heatmap(corr_ht,annot=True)
plt.show()

#statistical measurement of the data
print(data.describe)

#checking distribution of target variable
data['target'].value_counts()
sns.countplot(x='target', data=data)

y=data['target']
x=data.drop(columns='target',axis=1)
x=pd.get_dummies(x)
print(x)
print(y)

#splitting the data into train and test set
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test=train_test_split(x,y,test_size=0.2,random_state=2,stratify=y)

from sklearn.linear_model import LogisticRegression
lm=LogisticRegression()
lm.fit(x_train,y_train)
predicted_value=lm.predict(x_test)

from sklearn.metrics import confusion_matrix
confusion_matrix(y_test,predicted_value)

from sklearn.metrics import classification_report
print(classification_report(y_test, predicted_value))

from sklearn.metrics import accuracy_score
model = LogisticRegression()
model.fit(x_train, y_train)
# accuracy on training data
x_train_prediction = model.predict(x_train)
training_data_accuracy = accuracy_score(x_train_prediction, y_train)

#accuracy on test data
x_test_prediction = model.predict(x_test)
test_data_accuracy = accuracy_score(x_test_prediction, y_test)

print('Accuracy on Training data : ', training_data_accuracy)
print('Accuracy on Test data : ', test_data_accuracy)

#Accuracy on Training data :  0.8512396694214877
#Accuracy on Test data :  0.819672131147541