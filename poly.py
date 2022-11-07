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

'''
References

Used data from here:
# https://www.kaggle.com/datasets/sinakaraji/covid-vaccination-vs-death?resource=download

'''
