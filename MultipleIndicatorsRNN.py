# Recurrent Neural Network

# Part 1 - Data Preprocessing

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the training set
dataset_train = pd.read_csv('Google_Stock_Price_Train.csv')
cols = 4
training_set = dataset_train.iloc[:, 1:cols].values

# Feature Scaling
from sklearn.preprocessing import MinMaxScaler
sc = MinMaxScaler(feature_range = (0, 1))
training_set_scaled = sc.fit_transform(training_set)

# Creating a data structure with 60 timesteps and 1 output
X_train = []
y_train = []
days=60

for i in range(days, len(training_set_scaled)):
    X_train.append(training_set_scaled[i-days:i, :])
    y_train.append(training_set_scaled[i, 0])
X_train = np.array(X_train)
y_train = np.array(y_train)

# Part 2 - Building the RNN

# Importing the Keras libraries and packages
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Dropout

# Initialising the RNN
regressor = Sequential()

# Adding the first LSTM layer and some Dropout regularisation
regressor.add(LSTM(units = 50, return_sequences = True, input_shape = (X_train.shape[1], X_train.shape[2])))
regressor.add(Dropout(0.2))

# Adding a second LSTM layer and some Dropout regularisation
regressor.add(LSTM(units = 50, return_sequences = True))
regressor.add(Dropout(0.2))

# Adding a third LSTM layer and some Dropout regularisation
regressor.add(LSTM(units = 50, return_sequences = True))
regressor.add(Dropout(0.2))

# Adding a fourth LSTM layer and some Dropout regularisation
regressor.add(LSTM(units = 50))
regressor.add(Dropout(0.2))

# Adding the output layer
regressor.add(Dense(units = 1))
#regressor.add(Dense(units = 5))

# Compiling the RNN
regressor.compile(optimizer = 'adam', loss = 'mean_squared_error')

#for grid search i.e Parameter Tuning
'''regressor.compile(optimizer = optimizer, loss = 'mean_squared_error', metrics = ['accuracy'])
    return regressor
from keras.wrappers.scikit_learn import KerasRegressor    
regressor = KerasClassifier(build_fn = build_regressor)
parameters = {'batch_size': [32, 64],
              'epochs': [100, 150],
              'optimizer': ['adam', 'rmsprop']}
grid_search = GridSearchCV(estimator = regressor,
                           param_grid = parameters,
                           scoring = 'neg_mean_squared_error',
                           cv = 10)
grid_search = grid_search.fit(X_train, y_train)
best_parameters = grid_search.best_params_
best_accuracy = grid_search.best_score_'''

# Fitting the RNN to the Training set
regressor.fit(X_train, y_train, epochs = 2, batch_size = 32)

# Part 3 - Making the predictions and visualising the results

# Getting the real stock price of 2017
dataset_test = pd.read_csv('Google_Stock_Price_Test.csv')
real_stock_price = dataset_test.iloc[:, 1:2].values

# Getting the predicted stock price of 2017

dataset_total = dataset_train.append(dataset_test)
inputs = dataset_total.iloc[len(dataset_total) - len(dataset_test) - days:, 1:cols].values

inputs = sc.transform(inputs)

X_test = []
for i in range(days, days+len(dataset_test)):
    X_test.append(inputs[i-days:i, 0:3])
X_test = np.array(X_test)

predicted_stock_price = regressor.predict(X_test)
temp = np.zeros((len(predicted_stock_price), cols-1))
for i in range(0, len(predicted_stock_price)):
    temp[i,0]=predicted_stock_price[i, 0]
    
predicted_stock_price = sc.inverse_transform(temp)[:,0]
#predicted_stock_price = sc.inverse_transform(predicted_stock_price)

# Visualising the results
plt.plot(real_stock_price, color = 'red', label = 'Real Stock Price')
plt.plot(predicted_stock_price, color = 'blue', label = 'Predicted Stock Price')
plt.title('Stock Price Prediction')
plt.xlabel('Time')
plt.ylabel('Stock Price')
plt.legend()
plt.show()