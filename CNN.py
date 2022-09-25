from sklearn.preprocessing import MinMaxScaler
import numpy as np
import pandas as pd
import csv
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Flatten
from keras.layers.convolutional import Conv1D
from keras.layers.convolutional import MaxPooling1D

import util
import eval

np.random.seed(123)

lag = 1            # input dimension
outputDim = 1       # output dimension
rate = 0.76          # train rate
n_features = 1
hidden_dim = [8, 16, 32, 64, 100, 128, 256, 512]     # number of hidden units of MLP
epoch = 5
batch_size = 1
lr = 0.001           # learning rate
plot_flag = False

output = []

df = pd.read_csv("./data/port_data.csv")
df = df.fillna(0)
columns = df.columns

for i in range(len(hidden_dim)):
    print('hidden_dim i:', hidden_dim[i])

    for j in range(len(df.columns)-1):
        # ------------------------------------1. data preprocess-----------------------------------------------------
        # 1.1 load data
        # j = 0
        ts = df[columns[j+1]]
        data = ts.values.reshape(-1, 1).astype("float32")  # (N, 1)

        # 1.2 normalisation
        scaler = MinMaxScaler(feature_range=(0.0, 1.0)).fit(data)
        dataset = scaler.transform(data)

        # 1.3 divide the series into training/testing samples
        train_size = int(len(dataset) * rate)
        test_size = len(dataset) - train_size
        train, test = dataset[0:train_size], dataset[train_size:]

        trainX, trainY = util.createSamples(train, lag, RNN=False)
        testX, testY = util.createSamples(test, lag, RNN=False)

        # -------------------------------------------2. buil model and train--------------------------------------------
        # 2.1 build model
        model = Sequential()
        model.add(Conv1D(filters=64, kernel_size=1, activation='relu', input_shape=(lag, n_features)))
        model.add(MaxPooling1D(pool_size=1))
        model.add(Flatten())
        model.add(Dense(50, activation='relu'))
        model.add(Dense(lag))
        model.compile(optimizer='adam', loss='mse')

        # 2.2 train model
        trainX = trainX.reshape((trainX.shape[0], trainX.shape[1], n_features))
        model.fit(trainX, trainY, epochs=epoch, verbose=0)

        # --------------------------------------------3. forecasting-----------------------------------------------------
        # 3.1 get test results
        trainPred = model.predict(trainX)
        testPred = model.predict(testX)

        # 3.2 reverse the time series
        trainPred = scaler.inverse_transform(trainPred)
        trainY = scaler.inverse_transform(trainY)
        testPred = scaler.inverse_transform(testPred)
        testY = scaler.inverse_transform(testY)

        # ---------------------------------------------4. evaluate-------------------------------------------------------
        train_MAPE = eval.calcMAPE(trainY, trainPred)
        train_RMSE = eval.calcRMSE(trainY, trainPred)

        MAPE = eval.calcMAPE(testY, testPred)
        RMSE = eval.calcRMSE(testY, testPred)
        SMAPE = eval.calcSMAPE(testY, testPred)

        print('\n-----------------------------test----------------------------------')
        print('MAPE:', '\t\tRMSE', '\t\tSMAPE')
        print(MAPE, '\t', RMSE, '\t', SMAPE)

        if plot_flag:
            util.plot(trainPred, trainY, testPred, testY)

        # ----------------------------------------------5. save output---------------------------------------------------
        _output = [hidden_dim[i], columns[j+1], trainPred, trainY, train_MAPE, train_RMSE, testPred, testY, MAPE, RMSE]
        output.append(_output)

        f = open('output.csv', 'w', encoding='utf-8', newline='')
        csv_writer = csv.writer(f)
        header = ('hidden_size', 'port', 'train_fitted', 'train_real', 'train_MAPE', 'train_RMSE', 'test_results', 'test_real', 'test_MAPE', 'test_RMSE')
        csv_writer.writerow(header)
        for data in output:
            csv_writer.writerow(data)
        f.close()
