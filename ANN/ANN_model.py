# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


dataset = pd.read_csv('D:\Deep-learning\ANN\Churn_Modelling.csv')
x = dataset.iloc[:,3:13].values
y = dataset.iloc[:,13].values


from sklearn.preprocessing import LabelEncoder,OneHotEncoder
labelencoder_x_1 = LabelEncoder()
x[:,1] = labelencoder_x_1.fit_transform(x[:,1])
labelencoder_x_2 =  LabelEncoder()
x[:,2] = labelencoder_x_2.fit_transform(x[:,2])
onehotencoder = OneHotEncoder(categorical_features=[1])
x = onehotencoder.fit_transform(x).toarray()
x = x[:,1:]


from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=.25,random_state=0)


from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
x_train = sc.fit_transform(x_train)
x_test = sc.fit_transform(x_test)

import keras
from keras.models import Sequential
from keras.layers import Dense 

classifier = Sequential()

classifier.add(Dense(output_dim=6,init ="uniform",activation="relu",input_dim=11))
classifier.add(Dense(output_dim=6,init ="uniform",activation="relu"))
classifier.add(Dense(output_dim=1,init ="uniform",activation="sigmoid"))
classifier.compile(optimizer="adam",loss="binary_crossentropy",metrics=["accuracy"])
classifier.fit(x_train,y_train,batch_size=10,nb_epoch=100)

y_pred = classifier.predict(x_test)
y_pred = (y_pred >0.5)

from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test,y_pred)
percentage = (1926+211)/2500
print(percentage)

