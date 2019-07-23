# -*- coding: utf-8 -*-
"""
Created on Mon Jul 22 10:52:10 2019

@author: narendra.b.sinappa
"""

import pandas as pd

# CSV is available in the same folder in git, please change the path before you import the data.
data = pd.read_csv(r"C:\Users\narendra.b.sinappa\Desktop\heart.csv")


# separate independent and dependant variable
x = data.iloc[:,0:13].values
y = data.iloc[:,13].values.reshape(-1,1)

# train and test split (20% to test data)
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size = 0.2)


#scale or normalize the data to enhance the performance.
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
x_train = sc.fit_transform(x_train)
x_test =sc.fit_transform(x_test)


# model creation
from keras.models import Sequential
from keras.layers import Dense
cl = Sequential()

# adding the first hidden layer, where no of nodes = (input nodes+output nodes)/2
cl.add(Dense(output_dim = 6,init = "uniform", activation = "relu", input_dim = 13))

# second hidden layer
cl.add(Dense(output_dim = 6,init = "uniform", activation = "relu"))
cl.add(Dense(output_dim = 1,init = "uniform", activation = "sigmoid"))
cl.compile(optimizer = "adam", loss = "binary_crossentropy", metrics = ['accuracy'])
cl.fit(x_train,y_train, batch_size = 10, epochs = 200)

y_pred = cl.predict(x_test)
y_pred = (y_pred>0.8)

# confusion matrix to check the accuracy of a test data prediction
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test,y_pred)
