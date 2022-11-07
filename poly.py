# Imports
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures


# Functions
def read_csv():
    '''Reads the CSV file and returns as a pandas dataframe'''
    return pd.read_csv("covid-vaccination-vs-death_ratio.csv")


# Declares a new dataframe for storing UK data (1)
df = read_csv()
UK_df = df[df.country == "The United Kingdom"]

# Declares training data
train_x = np.asanyarray(UK_df[['ratio']])
train_y = np.asanyarray(UK_df[['New_deaths']])

# Creates a linear regression object
lin_reg = LinearRegression()

# Trains the Linear Regression Model
lin_reg.fit(train_x, train_y)
# Calculates the coefficients and intercepts for linear regression
XX = train_x
YY = lin_reg.intercept_[0] + lin_reg.coef_[0][0]*train_x

# Plots the graph
plt.plot(XX, YY, c='b', label="Linear Regression")



'''
References

Used data from here:
# https://www.kaggle.com/datasets/sinakaraji/covid-vaccination-vs-death?resource=download

(1) Used info from here on pandas dataframe for csv:
https://pandas.pydata.org/docs/reference/api/pandas.read_csv.html

(2) Used info from here on ~ in python:
https://blog.finxter.com/tilde-python/#:~:text=What%20is%20the%20Tilde%20~%20in,1%20and%20~101%20becomes%20010%20.

(3) Used info from here on numpy arange:
https://numpy.org/doc/stable/reference/generated/numpy.arange.html
'''
