# -*- coding: utf-8 -*-
"""
Created on Mon May 28 09:55:00 2018

@author: En-Chi
"""
#%%

### Part1 Data Preprocessing

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('Churn_Modelling.csv')
#%%
X = dataset.iloc[:, 3:13].values
y = dataset.iloc[:,13].values

#%%

# Encoding categorical data
# Encoding the Independent Variable
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder_X_1 = LabelEncoder()
X[:, 1] = labelencoder_X_1.fit_transform(X[:, 1])

#%%
labelencoder_X_2 = LabelEncoder()
X[:, 2] = labelencoder_X_2.fit_transform(X[:, 2])
#%%
onehotencoder = OneHotEncoder(categorical_features = [1])
X = onehotencoder.fit_transform(X).toarray()

X = X[:,1:]

# Splitting the dataset into the Training set and Test set
from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)
#%%
# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)
#%%

##### Part 2 -- Now let's make the ANN
###########################################

#%%
import keras
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
#%%

### Initialising the ANN
classifier = Sequential()

## Adding the input layer and first hidden layer with dropout
classifier.add(Dense(units = 6,kernel_initializer = 'uniform',activation = 'relu',input_dim = 11))
classifier.add(Dropout(0.1))

classifier.add(Dense(units = 6,kernel_initializer = 'uniform',activation = 'relu'))
classifier.add(Dropout(0.1))
classifier.add(Dense(units = 1,kernel_initializer = 'uniform',activation = 'sigmoid'))

classifier.compile(optimizer = 'adam',loss = 'binary_crossentropy',metrics = ['accuracy'])
#%%
classifier.fit(X_train,y_train,batch_size = 10,epochs = 100)


#%%



# Fitting classifier to the Training set
# Create your classifier here

# Predicting the Test set results
y_pred = classifier.predict(X_test)
y_pred = (y_pred > 0.5)
#%%
# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)



#%%

### Homework 1

customer_df = pd.DataFrame({"CreditScore":600,
                         "Geograpy":'France',"Gender":'Male',"Age":40,
                         "Tenure":3,"Balance":60000,"NumOfProducts":2,
                         "HasCrCard":1,"IsActiveMember":1,
                         "EstimatedSalary":50000},index = [0])

customer_df = customer_df[['CreditScore','Geograpy','Gender','Age','Tenure','Balance','NumOfProducts','HasCrCard',
                     'IsActiveMember','EstimatedSalary']]
#%%
customer = customer_df.values
customer[:, 1] = labelencoder_X_1.transform(customer[:, 1])
customer[:, 2] = labelencoder_X_2.transform(customer[:, 2])

#%%

customer = onehotencoder.transform(customer).toarray()

customer = customer[:,1:]
customer_std = sc.transform(customer)
#%%
customer_pred = classifier.predict(customer_std)
customer_pred = (customer_pred>0.5)
customer_pred
######################################################################


#%%

### Evaluate the ANN
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV
import keras
from keras.models import Sequential
from keras.layers import Dense
def build_classifier():
    classifier = Sequential()

    ## Adding the input layer and first hidden layer
    classifier.add(Dense(units = 6,kernel_initializer = 'uniform',activation = 'relu',input_dim = 11))
    classifier.add(Dense(units = 6,kernel_initializer = 'uniform',activation = 'relu'))
    
    classifier.add(Dense(units = 1,kernel_initializer = 'uniform',activation = 'sigmoid'))
    
    classifier.compile(optimizer = 'adam',loss = 'binary_crossentropy',metrics = ['accuracy'])
    
    return classifier

classifier = KerasClassifier(build_fn = build_classifier)


parameters = {'batch_size':[25,32],
              'nb_epoch':[100,500],
              'optimizer':['adam','rmsprop']}

grid_search = GridSearchCV(estimator = classifier,
                           param_grid = parameters, 
                           scoring = 'accuracy',
                           cv = 10)

grid_search.fit(X_train,y_train)
best_parameters = grid_search.best_params_
best_accuracy = grid_search.best_score_ 


#%%
mean = accuracies.mean()
variance = accuracies.std() 

#%%