# GRIPTASK1
Linear Regression with Python Scikit Learn

#importing  important libaries in python to perform task
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
#reading data from given url link in task
data="http://bit.ly/w-data"
df=pd.read_csv(data)
print("data has been succesfully imported\n")
df
#distribution of scores
x='Hours'
y='Scores'

df.plot(x,y,style='o')
plt.title('hours vs percentage')
plt.xlabel('hours studied')
plt.ylabel('percentage scored')
plt.grid()
plt.show
#next step is to divide the data into "attributes" (inputs) and "labels" (outputs).
X = df.iloc[:, :-1].values
y = df.iloc[:, 1].values
#divide into training and test sets by using attributes and labels.We'll do this by using Scikit-Learn's built-in train_test_split() method:
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
from sklearn.linear_model import LinearRegression

regressor = LinearRegression()
regressor.fit(X_train, y_train)

print("Model is trained successfully")
#plotting the regreession line
# Plotting the regression line
line = regressor.coef_ * X + regressor.intercept_

# Plotting for the test data
plt.scatter(X, y)
plt.plot(X, line)
plt.show()
#To make predictions
print(X_test)
y_pred = regressor.predict(X_test)
df = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred})  
df 
#What will be predicted score if a student studies for 9.25 hrs/ day?
#Predicting with dataset
hours = 9.25
own_pred = regressor.predict([[hours]])
print("No of Hours = {}".format(hours))
print("Predicted Score = {}".format(own_pred[0]))
from sklearn import metrics  
print('Mean Absolute Error:', metrics.mean_absolute_error(y_test, y_pred)) 
