#!/usr/bin/env python
# coding: utf-8

# ## 第零部分 初始化

# In[2]:


# import packages
import pandas as pd
import numpy as np

# to plot within notebook
import matplotlib.pyplot as plt
#get_ipython().run_line_magic('matplotlib', 'inline')

# setting figure size
from matplotlib.pylab import rcParams
rcParams['figure.figsize'] = 20,10

# for normalizing data
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler(feature_range=(0, 1))


# In[3]:


# read the file
df = pd.read_csv('../res/data_meg.csv')

# print the head
df.head()


# In[3]:


# setting index as date
# df['Ndate'] = pd.to_datetime(df.Ndate,format='%Y-%m-%d')
# df.index = df['Ndate']

# plot
# plt.figure(figsize=(16,8))
# plt.plot(df['gold'], label='Price History')


# In[4]:


# splitting into train and validation
# train = df[:5500].copy()
# valid = df[5500:].copy()
#
# df.shape, train.shape, valid.shape
#

# # ## 第一部分 滑动平均
#
# # In[44]:
#
#
# #make predictions
# preds = []
# for i in range(0,1277):
#     a = train['gold'][len(train)-1277+i:].sum() + sum(preds)
#     b = a/1277
#     preds.append(b)
# preds
#
#
# # In[45]:
#
#
# #calculate rmse
# rms=np.sqrt(np.mean(np.power((np.array(valid['gold'])-preds),2)))
# rms
#
#
# # In[46]:
#
#
# #plot
# valid['Predictions'] = 0
# valid['Predictions'] = preds
# plt.plot(train['gold'])
# plt.plot(valid[['gold', 'Predictions']])
#
#
# # ## 第二部分 长短期记忆网络(LSTM)
#
# # In[4]:
#
#
# #importing required libraries
# from sklearn.preprocessing import MinMaxScaler
# from keras.models import Sequential
# from keras.layers import Dense, Dropout, LSTM
#
#
# # In[5]:
#
#
# #creating dataframe
# data = df.sort_index(ascending=True, axis=0)
# new_data = pd.DataFrame(index=range(0,len(df)),columns=['gold'])
# for i in range(0,len(data)):
#     new_data['gold'][i] = data['gold'][i]
#
#
# # In[6]:
#
#
# #creating train and test sets
# dataset = new_data.values
# train = dataset[0:5500,:]
# valid = dataset[5500:,:]
#
# #converting dataset into x_train and y_train
# scaler = MinMaxScaler(feature_range=(0, 1))
#
#
# # In[7]:
#
#
# scaled_data = scaler.fit_transform(dataset)
# x_train, y_train = [], []
#
# for i in range(60,len(train)):
#     x_train.append(scaled_data[i-60:i,0])
#     y_train.append(scaled_data[i,0])
#
# x_train, y_train = np.array(x_train), np.array(y_train)
# x_train = np.reshape(x_train, (x_train.shape[0],x_train.shape[1],1))
#
#
# # In[8]:
#
#
# # create and fit the LSTM network
# model = Sequential()
# model.add(LSTM(units=50, return_sequences=True, input_shape=(x_train.shape[1],1)))
# model.add(LSTM(units=50))
# model.add(Dense(1))
# model.compile(loss='mean_squared_error', optimizer='adam')
# model.fit(x_train, y_train, epochs=1, batch_size=1, verbose=2)
#
#
# # In[9]:
#
#
# #predicting 246 values, using past 60 from the train data
# inputs = new_data[len(new_data) - len(valid) - 60:].values
# inputs = inputs.reshape(-1,1)
# inputs  = scaler.transform(inputs)
#
# X_test = []
#
# for i in range(60,inputs.shape[0]):
#     X_test.append(inputs[i-60:i,0])
#
# X_test = np.array(X_test)
#
# X_test = np.reshape(X_test, (X_test.shape[0],X_test.shape[1],1))
# closing_price = model.predict(X_test)
# closing_price = scaler.inverse_transform(closing_price)
#
#
# # In[10]:
#
#
# rms=np.sqrt(np.mean(np.power((valid-closing_price),2)))
# rms
#
#
# # In[11]:
#
#
# #for plotting
# train = new_data[:5500].copy()
# valid = new_data[5500:].copy()
# valid['Predictions'] = closing_price
# plt.plot(train['gold'])
# plt.plot(valid[['gold','Predictions']])
#
#
# # In[13]:
#
#
# type(valid)
#
#
# # In[14]:
#
#
# valid.to_csv('LSTM_predict.csv')
#

# ## 第三部分 神经网络

# In[3]:


import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from keras.utils import np_utils
import keras as k
df = pd.read_csv('../res/data_meg.csv')


# In[4]:


df


# In[10]:


Y_PRED = []
Y_TEST = []
accuracy = []
precision = []
recall = []
f1 = []
df = df.reindex(np.random.permutation(df.index))
# 黄金价格涨跌放在最后一列，列名叫‘Target’，涨和不变记为1，跌记为0
x = df.iloc[:, :-1]
y = df.iloc[:, -1]


# In[11]:


# 填参数的数目
num_parameters = 3


# In[12]:


# 十折
sfolder = StratifiedKFold(n_splits=10, shuffle=True, random_state=1337)
for train, test in sfolder.split(x, y):
    batch_size = 64
    max_epochs = 12
    model = k.models.Sequential()
    # 每层神经元数可调
    model.add(k.layers.Dense(units=32, activation='relu', use_bias=True, input_shape=(num_parameters, )))
    model.add(k.layers.normalization.BatchNormalization(epsilon=1e-6))
    model.add(k.layers.Dense(units=32, activation='relu', use_bias=True))
    model.add(k.layers.normalization.BatchNormalization())
    model.add(k.layers.Dense(units=32, activation='relu', use_bias=True))
    model.add(k.layers.normalization.BatchNormalization(epsilon=1e-6))
    model.add(k.layers.Dense(units=32, activation='relu', use_bias=True))
    model.add(k.layers.normalization.BatchNormalization())
    model.add(k.layers.Dropout(0.5))
    model.add(k.layers.Dense(units=2, activation='sigmoid'))
    model.compile(loss='mean_squared_error', optimizer='adam', metrics=['mae'])
    x_train = x.iloc[train.tolist(), :]
    y_train = y.iloc[train.tolist()]
    model.fit(x_train, np_utils.to_categorical(y_train, num_classes=2), batch_size=batch_size, epochs=max_epochs, shuffle=True, verbose=1)
    x_test = x.iloc[test.tolist(), :]
    y_test = y.iloc[test.tolist()]
    y_test = y_test.reset_index()['Target']
    y_pred = model.predict(x_test)
    test_0_num = 0
    test_0_pred_0 = 0
    test_1_num = 0
    test_1_pred_1 = 0
    for index in range(0, len(y_test)):
        if y_test[index] == 0:
            test_0_num = test_0_num + 1
            if y_pred[index][0] < 0.5:
                test_0_pred_0 = test_0_pred_0 + 1
        else:
            test_1_num = test_1_num + 1
            if y_pred[index][1] > 0.5:
                test_1_pred_1 = test_1_pred_1 + 1
    accuracy = accuracy + [(test_1_pred_1 + test_0_pred_0) / (test_0_num + test_1_num)]
    precision = precision + [test_1_pred_1 / (test_1_pred_1 + test_0_num - test_0_pred_0)]
    recall = recall + [test_1_pred_1 / (test_1_pred_1 + test_0_pred_0)]


# In[13]:


[accuracy, precision, recall]


# In[ ]:




