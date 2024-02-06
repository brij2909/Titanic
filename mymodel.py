# Import Libraries

import pandas as pd
import numpy as np
import random as rnd
import seaborn as sbn
import sys
import matplotlib.pyplot as plt


# Import Machine Learning Liabraries

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC, LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import RandomForestRegressor
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neighbors import KNeighborsRegressor
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import Perceptron
from sklearn.linear_model import SGDClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import DecisionTreeRegressor

# Data Collection & Processing
my_model = pd.read_csv(r"C:\Users\Dell\Desktop\Machine Learning\My Model\Titanic\tit_train.csv")
print(my_model.head())
print('----------------------------------------------------------------')
print(my_model.shape)
print('----------------------------------------------------------------')
print(my_model.info())
print('----------------------------------------------------------------')
# Check the number of missing values
print(my_model.isnull().sum())
print('----------------------------------------------------------------')
# Handling the missing values
my_model = my_model.drop(columns='Cabin', axis=1)
print('----------------------------------------------------------------')
# Replacing the missing values in 'Age' column with mean value
my_model['Age'].fillna(my_model['Age'].mean(), inplace=True)
print('----------------------------------------------------------------')
# Finding the mode value of 'Enbarked' Column
print(my_model['Embarked'].mode())
print('----------------------------------------------------------------')
print(my_model['Embarked'].mode()[0])
print('----------------------------------------------------------------')
# Replacing the missing values in 'Embarked' column with mode value
my_model['Embarked'].fillna(my_model['Embarked'].mode()[0], inplace=True)
print('----------------------------------------------------------------')
# Check the number of missing values in each column
print(my_model.isnull().sum())
print('----------------------------------------------------------------')

import seaborn as sbn
import matplotlib.pyplot as plt

# Assuming 'Survived' is a column in 'my_model' DataFrame
sbn.countplot(x='Survived', data=my_model)
plt.show()
print('----------------------------------------------------------------')
# Data Analysis
# Get some stastical measures about the data
print(my_model.describe())
print('----------------------------------------------------------------')
# Finding the number of people survived or not survived
print(my_model['Survived'].value_counts())
print('----------------------------------------------------------------')
# Enconding the categorical data
print(my_model['Sex'].value_counts())
print('----------------------------------------------------------------')
print(my_model['Embarked'].value_counts())
print('----------------------------------------------------------------')
# Converting Cetegorical column
my_model.replace({'Sex':{'male':1, 'female':0}, 'Embarked':{'S':0, 'C':1, 'Q':2}}, inplace=True)
print(my_model.head())
print('----------------------------------------------------------------')
# Seperating Featured & Target
X = my_model.drop(columns=['PassengerId', 'Name', 'Ticket', 'Survived'], axis=1)
Y = my_model['Survived']
print(X)
print('----------------------------------------------------------------')
print('----------------------------------------------------------------')
print(Y)
print('----------------------------------------------------------------')
# Splitting the data into training data and test data
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=2)
print(X.shape, X_train.shape, X_test.shape)
print('----------------------------------------------------------------')
# Model Training
# Logistic Regression
# training the Logistic Regression model with training data
model = LogisticRegression()
model.fit(X_train, Y_train)
print('----------------------------------------------------------------')
# Model Evaluation
# Accuracy Score
# Accuracy on training data
X_train_prediction = model.predict(X_train)
print(X_train_prediction)
print('----------------------------------------------------------------')
train_data_accuracy = accuracy_score(Y_train, X_train_prediction)*100
print('Accuracy Score Of Train Data is ', train_data_accuracy)
print('----------------------------------------------------------------')
# Accuracy On test data
X_test_prediction = model.predict(X_test)
print(X_test_prediction)
test_data_accuracy = accuracy_score(Y_test, X_test_prediction)*100
print('Accuracy Score Of Test Data is ', test_data_accuracy)
print('----------------------------------------------------------------')
import pickle

# Assuming 'model' is your trained machine learning model
save_mymodel = pickle.dumps(model)

# Save the model as a pickle file
with open('model.pkl', 'wb') as file:
    file.write(save_mymodel)

# Load the model from the pickle file
with open('model.pkl', 'rb') as file:
    loaded_model = pickle.loads(file.read())

# Assuming 'X_test' is your test data
predictions = loaded_model.predict(X_test)

from joblib import dump, load

# Assuming 'model' is your trained machine learning model
dump(model, 'model.joblib')

# Load the model from the joblib file
loaded_model = load('model.joblib')

# Assuming 'X_test' is your test data
predictions = loaded_model.predict(X_test)