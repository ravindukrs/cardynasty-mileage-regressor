import math
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import csv
from sklearn.preprocessing import LabelEncoder
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_predict, cross_validate
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.svm import SVR
import pickle


#Read Data
dataset = pd.read_csv('milagedata.csv')

#Define X and Y columns
X = dataset.iloc[:,[0,1]].values
Y = dataset.iloc[:, 2].values #2 for Mileage

#Encode 'Body Type'
le_X_0= LabelEncoder()
X[:, 0] = le_X_0.fit_transform(X[:, 0])

#Create Scaler
scaler = MinMaxScaler(feature_range=(0, 1))

#Apply Scaler on X

scaler.fit(X)
X = scaler.transform(X)

#Convert Y to 1D Array - Not necessary
Y = Y.flatten()

#Shuffle Data
X, Y = shuffle(X, Y, random_state=42)

#Split Training and Testing Data
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size = 0.2, random_state = 1)

#Get Best parameters
gsc = GridSearchCV(
    estimator=SVR(kernel='rbf'),
    param_grid={
      'C': [0.1, 1, 100, 1000],
      'epsilon': [0.0001, 0.0005, 0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1, 5, 10],
      'gamma':[0.0001, 0.001, 0.002, 0.005, 0.1, 0.2]
    },
    cv=5, scoring='neg_mean_squared_error', verbose=0, n_jobs=-1
)

grid_result = gsc.fit(X_train, y_train)
best_params = grid_result.best_params_
print("\nBest: %f using %s" % (grid_result.best_score_, grid_result.best_params_))

rbf_svr = SVR(
    kernel='rbf',
    C=best_params["C"],
    epsilon=best_params["epsilon"],
    gamma=best_params["gamma"],
    coef0=0.1,
    shrinking=True,
    tol=0.001,
    cache_size=200,
    verbose=False,
    max_iter=-1
)

rbf_svr.fit(X_train, y_train)

y_test_predictions = rbf_svr.predict(X_test)
y_train_predictions = rbf_svr.predict(X_train)

mse = mean_squared_error(y_test, y_test_predictions, squared=True)
rmse = mean_squared_error(y_test, y_test_predictions, squared=False)

print("MSE: ",mse,"\n","RMSE: ",rmse,"\n")

pickle.dump(rbf_svr, open('svrmodel.sav', 'wb'))
pickle.dump(scaler, open('scaler.sav', 'wb'))
pickle.dump(le_X_0, open('label_encoder.sav', 'wb'))
