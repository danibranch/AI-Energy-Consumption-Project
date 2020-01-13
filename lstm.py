import pandas as pd
from matplotlib import pyplot
from numpy import concatenate
from pandas import DataFrame, concat
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import LSTM
from tensorflow.keras.models import Sequential

import proiect


# convert series to supervised learning
def series_to_supervised(data, n_in=1, n_out=1, dropnan=True):
    n_vars = 1 if type(data) is list else data.shape[1]
    df = DataFrame(data)
    cols, names = list(), list()
    # input sequence (t-n, ... t-1)
    for i in range(n_in, 0, -1):
        cols.append(df.shift(i))
        names += [('var%d(t-%d)' % (j + 1, i)) for j in range(n_vars)]
    # forecast sequence (t, t+1, ... t+n)
    for i in range(0, n_out):
        cols.append(df.shift(-i))
        if i == 0:
            names += [('var%d(t)' % (j + 1)) for j in range(n_vars)]
        else:
            names += [('var%d(t+%d)' % (j + 1, i)) for j in range(n_vars)]
    # put it all together
    agg = concat(cols, axis=1)
    agg.columns = names
    # drop rows with NaN values
    if dropnan:
        agg.dropna(inplace=True)
    return agg


scaler = MinMaxScaler(feature_range=(0, 1))
train = pd.read_csv('dataset/train_electricity.csv', header=0, index_col=0)
train = proiect.rearrange_cols(train)
train = proiect.remove_outliers(train)
scaled = scaler.fit_transform(train.values)
reframed = series_to_supervised(scaled, 1, 1)
reframed.drop(reframed.columns[[8, 9, 10, 11, 12, 13, 14, 15, 16]], axis=1, inplace=True)
train = reframed.values

test = pd.read_csv('dataset/test_electricity.csv', header=0, index_col=0)
scaled = scaler.fit_transform(test.values)
reframed = series_to_supervised(scaled, 1, 1)
# TODO: Modify to get only the desired columns
reframed.drop(reframed.columns[[8, 9, 10, 11, 12, 13, 14]], axis=1, inplace=True)
test = reframed.values

# split into input and outputs
train_X, train_y = train[:, :-1], train[:, -1]
test_X, test_y = test[:, :-1], test[:, -1]
# reshape input to be 3D [samples, timesteps, features]
train_X = train_X.reshape((train_X.shape[0], 1, train_X.shape[1]))
test_X = test_X.reshape((test_X.shape[0], 1, test_X.shape[1]))
print(train_X.shape, train_y.shape, test_X.shape, test_y.shape)
# design network

model = Sequential()
model.add(LSTM(50, input_shape=(train_X.shape[1], train_X.shape[2])))
model.add(Dense(1))
model.compile(loss='mae', optimizer='adam')
model.load_weights('cp')
# fit network
history = model.fit(train_X, train_y, epochs=50, batch_size=72, verbose=2,
                    shuffle=False)
model.save_weights('cp')
# plot history
pyplot.plot(history.history['loss'], label='train')
pyplot.legend()
pyplot.show()

# make a prediction
yhat = model.predict(test_X)
test_X = test_X.reshape((test_X.shape[0], test_X.shape[2]))
# invert scaling for forecast
inv_yhat = concatenate((yhat, test_X[:, 1:]), axis=1)
inv_yhat = scaler.inverse_transform(inv_yhat)
inv_yhat = inv_yhat[:, 0]
# invert scaling for actual
test_y = test_y.reshape((len(test_y), 1))
inv_y = concatenate((test_y, test_X[:, 1:]), axis=1)

inv_y = scaler.inverse_transform(inv_y)
inv_y = inv_y[:, 0]
# calculate MSE
mse = mean_squared_error(inv_y, inv_yhat)
print('Test MSE: %.3f' % mse)

pyplot.plot(inv_yhat, label='predicted')
pyplot.plot(inv_y, label='actual')
pyplot.legend()
pyplot.show()
