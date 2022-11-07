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

# Divides data into test and training (2)
df = np.random.rand(len(UK_df)) < .8
train = UK_df[df]
test = UK_df[~df]

# Identify the dependent(y) and independent variables(x) in the train and test
# dataframes
train_x = np.asanyarray(train[['ratio']])
train_y = np.asanyarray(train[['New_deaths']])
test_x = np.asanyarray(train[['ratio']])
test_y = np.asanyarray(train[['New_deaths']])

# Generate polynomial and interaction features Object with 6 degrees
poly = PolynomialFeatures(degree=6)

# Makes a number of variables with different degrees from
# independent variables(x) to use them in a model
train_x_poly = poly.fit_transform(train_x)

# Make the model
lin_reg = LinearRegression()
train_y_ = lin_reg.fit(train_x_poly, train_y)

# Constructs a scatterplot using train data in yellow
plt.scatter(train.ratio, train.New_deaths,  color='y', label='Training data')
# Set the X axis using numpy:   np.arange(start, end, interval) (3)
XX = np.arange(train_x[0], train_x[-1], 0.1)
# Set the Y axis using intercept and coefficients
YY = lin_reg.intercept_[0]
for d in range(1, 7):
    YY += lin_reg.coef_[0][d]*np.power(XX, d)

# Plots regression model
plt.plot(XX, YY, 'r', label='Polynomial (6) Regression')
plt.title('COVID-19 Deaths Regressed on Vaccination Rate (%) in the United Kingdom')
plt.xlabel("Vaccination rate (%) ")
plt.ylabel("New deaths")
plt.grid(True)
plt.legend()
plt.show()



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
