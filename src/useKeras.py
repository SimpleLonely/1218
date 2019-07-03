from keras.models import Sequential
from keras.layers import Dense
import keras
import pandas as pd
import numpy as np

def normalize_cols(m):
    col_max = m.max(axis=0)
    col_min = m.min(axis=0)
    return (m-col_min)/(col_max-col_min)

data = pd.read_csv('C://WorkSpace//double-innovation//res//input0701.csv')
n = data.shape[0]
m = data.shape[1]
train_start = 1
train_end = int(np.floor(0.8 * n))
test_start = train_end + 1
test_end = n
data_train = data.loc[train_start: train_end]
data_test = data.loc[test_start: test_end]
x_data = data.loc[1:n].ix[:, 2:]
y_data = data.loc[1:n].ix[:, 0]
x_train = data_train.ix[:, 2:]
y_train = data_train.ix[:, 0]
x_test = data_test.ix[:, 2:]
y_test = data_test.ix[:, 0]
x_data = np.nan_to_num(normalize_cols(x_data))
x_test = np.nan_to_num(normalize_cols(x_test))
x_train = np.nan_to_num(normalize_cols(x_train))

model = Sequential()
model.add(Dense(units=64, activation='relu', input_dim=31))
model.add(Dense(units=64, activation='relu', input_dim=64))
model.add(Dense(units=1, activation='softmax')) 
model.compile(optimizer='rmsprop',
              loss='mse')
# x_train 和 y_train 是 Numpy 数组 -- 就像在 Scikit-Learn API 中一样。
model.fit(x_train, y_train, epochs=5, batch_size=32)
loss_and_metrics = model.evaluate(x_test, y_test, batch_size=128)