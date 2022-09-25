# -*- coding: utf-8 -*-
"""
univariate multi-step vector-output 1d cnn example

Created on Fri Feb 28 00:18:30 2020
@author: SalmanKarim
"""


from numpy import array
import numpy as np
import pandas as pd
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Flatten
from keras.layers.convolutional import Conv1D
from keras.layers.convolutional import MaxPooling1D
from typing import List
import eval
import csv


def split_sequence(sequence, n_steps_in, n_steps_out):
    X, y = list(), list()
    for i in range(len(sequence)):
        # find the end of this pattern
        end_ix = i + n_steps_in
        out_end_ix = end_ix + n_steps_out
        # check if we are beyond the sequence
        if out_end_ix > len(sequence):
            break
        # gather input and output parts of the pattern
        seq_x, seq_y = sequence[i:end_ix], sequence[end_ix:out_end_ix]
        X.append(seq_x)
        y.append(seq_y)
    return array(X), array(y)


# -----------------------------------------------1.  data preprocess--------------------------------------------
output = []

np.random.seed(123)
n_features = 1
n_steps_in = 1
n_steps_out = 1
epoch = 5000
n_filters = [8, 16, 32, 64, 100, 128, 256, 512]     # number of hidden units of MLP
rate = 0.76

df = pd.read_csv("./data/port_data.csv")
df = df.fillna(0)
columns = df.columns
for ii in range(len(n_filters)):
    print('hidden_dim i:', n_filters[ii])

    for j in range(len(df.columns) - 1):
        print('    ', columns[j+1])
        # define input sequence
        raw_seq = df[columns[j+1]]

        # split a univariate sequence into samples
        X, y = split_sequence(raw_seq, n_steps_in, n_steps_out)

        train_size = int(len(X) * rate)
        test_size = len(X) - train_size

        train_x = X[0:train_size]
        test_x = X[train_size:]

        train_y = y[0:train_size]
        test_y = y[train_size:]

        # reshape from [samples, timesteps] into [samples, timesteps, features]
        train_x = train_x.reshape((train_x.shape[0], train_x.shape[1], n_features))

        # --------------------------------------------------2. define model--------------------------------------------------
        model = Sequential()
        model.add(Conv1D(filters=n_filters[ii], kernel_size=1, activation='relu', input_shape=(n_steps_in, n_features)))
        model.add(MaxPooling1D(pool_size=1))
        model.add(Flatten())
        model.add(Dense(50, activation='relu'))
        model.add(Dense(n_steps_out))
        model.compile(optimizer='adam', loss='mse')

        # --------------------------------------------------3. fit model ----------------------------------------------------
        model.fit(train_x, train_y, epochs=epoch, verbose=0)

        train_fitted = []
        for i in train_x:
            x_input = array([i])
            x_input = x_input.reshape((1, n_steps_in, n_features))
            yhat = model.predict(x_input, verbose=0)
            train_fitted.append(yhat.tolist())

        trainFitted = np.ravel(train_fitted)
        trainFitted = pd.DataFrame(trainFitted).values.reshape(-1, 1)
        train_y = pd.DataFrame(train_y).values.reshape(-1, 1)

        train_MAPE = eval.calcMAPE(train_y, trainFitted)
        train_RMSE = eval.calcRMSE(train_y, trainFitted)

        # --------------------------------------------------4. evaluate -----------------------------------------------------
        forecast_result = []
        for i in test_x:
            x_input = array([i])
            x_input = x_input.reshape((1, n_steps_in, n_features))
            yhat = model.predict(x_input, verbose=0)
            forecast_result.append(yhat.tolist())

        testPred = np.ravel(forecast_result)    # np.ravel() 将list中的list去除
        testPred = pd.DataFrame(testPred).values.reshape(-1, 1)
        test_y = pd.DataFrame(test_y).values.reshape(-1, 1)

        test_MAPE = eval.calcMAPE(test_y, testPred)
        test_RMSE = eval.calcRMSE(test_y, testPred)
    # --------------------------------------------------5. save output---------------------------------------------------
        _output = [n_filters[ii], columns[j+1], trainFitted, train_y, train_MAPE, train_RMSE, testPred, test_y, test_MAPE, test_RMSE]
        output.append(_output)

f = open('output.csv', 'w', encoding='utf-8', newline='')
csv_writer = csv.writer(f)
header = ('n_filters', 'port', 'train_fitted', 'train_real', 'train_MAPE', 'train_RMSE', 'test_results', 'test_real', 'test_MAPE', 'test_RMSE')
csv_writer.writerow(header)
for data in output:
    csv_writer.writerow(data)
f.close()
