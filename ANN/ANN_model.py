# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
dataset = pd.read_csv('D:\Deep-learning\ANN\Churn_Modelling.csv')
x = dataset.iloc[:,3:13].values
y = dataset.iloc[:,13].values

x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=.25,random_state=0)

