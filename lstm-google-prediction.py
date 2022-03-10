# RNN stock price prediction

### Data preprocessing
# Import libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# import training set
dataset_train = pd.read_csv("./Google_Stock_Price_Train.csv")
training_set = dataset_train.iloc[:, 1:2].values

# feature scaling
from sklearn.preprocessing import MinMaxScaler
sc = MinMaxScaler(feature_range = (0, 1))
training_set_scaled = sc.fit_transform(training_set)

# Create data structure to hold x number of timesteps and 1 output 
# which means it checks x steps in the past to predict the next step
timesteps = 60

X_train = []
y_train = []

for i in range(timesteps, len(training_set)):
    X_train.append(training_set_scaled[i - timesteps:i, 0])
    y_train.append(training_set_scaled[i, 0])

X_train, y_train = np.array(X_train), np.array(y_train)

# Reshape X_train to add new dimension
# This step is needed because keras needs a 3D tensor as input
# [batch, timesteps, feature]
# Read more at https://keras.io/api/layers/recurrent_layers/lstm/
X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))

### Build RNN

# Import libraries
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import LSTM
from tensorflow.keras.layers import Dropout

print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))

# initialize RNN as a sequence of layers
regressor = Sequential()

# add LSTM layers
# first LSTM layer
regressor.add(LSTM(units = 100, return_sequences = True, input_shape = X_train.shape[1:]))
regressor.add(Dropout(0.2))

# second LSTM layer
regressor.add(LSTM(units = 100, return_sequences = True))
regressor.add(Dropout(0.2))

# third LSTM layer
regressor.add(LSTM(units = 100, return_sequences = True))
regressor.add(Dropout(0.2))

# fourth LSTM layer
regressor.add(LSTM(units = 100))
regressor.add(Dropout(0.2))

# Add output layer
regressor.add(Dense(units = 1))

# Compile and choose optimizer (RMSprop and Adam are preferred) as well as
# loss function, which mean squared error is ideal for this case scenario
regressor.compile(optimizer = "adam", loss = 'mean_squared_error')

### Train RNN
regressor.fit(X_train, y_train, epochs = 100, batch_size = 32)

### PREDICT

# Get real stock prices
dataset_test = pd.read_csv("./Google_Stock_Price_Test.csv")
real_stock_price = dataset_test.iloc[:, 1:2].values

# combine datasets

dataset_total = pd.concat((dataset_train['Open'], dataset_test['Open']), axis = 0)

# generate the inputs (with this we get a normal array)
inputs = dataset_total.iloc[len(dataset_total) - len(dataset_test) - timesteps:].values

# convert this to a numpy array (matrix)
inputs = inputs.reshape(-1, 1)

# escalate dataset
inputs = sc.transform(inputs)

# create 3d tensor for predict
X_test = []

for i in range(timesteps, len(inputs)):
    X_test.append(inputs[i - timesteps:i, 0])

X_test = np.array(X_test)

X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))

# predict

prediction = regressor.predict(X_test)

# inverse transform results

prediction = sc.inverse_transform(prediction)

import math
from sklearn.metrics import mean_squared_error
rmse = math.sqrt(mean_squared_error(real_stock_price, prediction))
print(rmse)

### visualize results

plt.plot(real_stock_price, color = 'red')
plt.plot(prediction, color = 'blue')
plt.title('Google stock price prediction')
plt.xlabel('Time')
plt.ylabel('Price')
plt.legend()
plt.show()

