import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split

from sklearn.linear_model import LinearRegression

from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

import matplotlib.pyplot as plt
import seaborn as sns

import pickle
import json

df = pd.read_csv('medical_insurance.csv')

x = df.drop('charges',axis = 1)
y = df['charges']

x_train, x_test, y_train, y_test = train_test_split(x,y, test_size=0.2, random_state=15)

lin_reg_model = LinearRegression()
lin_reg_model.fit(x_train, y_train)