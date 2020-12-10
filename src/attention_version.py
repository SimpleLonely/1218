# -*- coding: utf-8 -*-
"""
Created on Thu Dec 10 01:55:28 2020

@author: hxh85
"""


#%%


import warnings

import numpy as np 
import pandas as pd 
import tensorflow as tf
import matplotlib.pyplot as plt

warnings.filterwarnings('ignore')


#%%

from tensorflow.keras import backend as K
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Dropout, LSTM, Lambda, Multiply

from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, f1_score, recall_score


#%%


config = tf.compat.v1.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.compat.v1.Session(config=config)

scaler = StandardScaler()
yscaler = StandardScaler()


#%%


close_df = pd.read_csv('../res/close_df.csv', index_col=0)
df = np.log(close_df).diff()
target = df['XAU'].iloc[2:].values
features = df.iloc[1:-1].values


#%%


train_val_split = 6000
train_test_split = 7500

X_train = scaler.fit_transform(features[:train_val_split])
y_train = target[:train_val_split].reshape(-1,1)

X_val = scaler.transform(features[train_val_split:train_test_split])
y_val = target[train_val_split:train_test_split].reshape(-1,1)

X_test = scaler.transform(features[train_test_split:])
y_test = target[train_test_split:].reshape(-1,1)


#%%


print(X_train.shape, y_train.shape, X_test.shape, y_test.shape)


#%%


# act = 'tanh'
# 损失函数考虑修改：加mse正则项
def return_loss(y_true, y_pred):
    # pos = K.sign(y_pred)
    loss = -K.mean(y_pred * y_true)
    return loss


#%%


INPUT_DIM = 13
OUTPUT_DIM = 1
batch_size = 128 
epochs = 30
 
seq_len = 5
hidden_size = 128


#%%


inputs = Input(shape=(INPUT_DIM))

x = Dense(64, activation='tanh', name='layer_dense')(inputs)
x = Dropout(0.2)(x)
x = Lambda(lambda x:K.expand_dims(x))(x)
lstm_out = LSTM(32, activation='tanh', name='layer_lstm')(x)

attention_probs = Dense(32, activation='tanh', name='attention_probs')(lstm_out)
attention_mul = Multiply()([lstm_out, attention_probs])

output = Dense(1, activation='tanh')(attention_mul)

model = Model(inputs=inputs, outputs=output)
print(model.summary())


#%%


model.compile(loss='mean_squared_error', optimizer='adam')
rs = model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, shuffle=False)


#%%


y_pred = model.predict(X_test)
print('MSE Train loss:', model.evaluate(X_train, y_train, batch_size=batch_size))
print('MSE Test loss:', model.evaluate(X_test, y_test, batch_size=batch_size))
plt.plot(y_test, label='test')
plt.plot(y_pred, label='pred')
plt.legend()
plt.show()


#%%


y_true = np.sign(y_train)
y_true[y_true==-1] = 0
y_pred = np.sign(model.predict(X_train))
y_pred[y_pred==-1] = 0

print(accuracy_score(y_pred, y_true), precision_score(y_true, y_pred), recall_score(y_true, y_pred), f1_score(y_true, y_pred))
