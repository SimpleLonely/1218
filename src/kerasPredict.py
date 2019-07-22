import keras
import pandas as pd
import numpy as np
from keras.models import load_model
import matplotlib.pyplot as plt
data = pd.read_csv('C://WorkSpace//double-innovation//res//input0701.csv')
n = data.shape[0]
m = data.shape[1]
x_data = data.loc[1:n].ix[:, 2:]
y_data = data.loc[1:n].ix[:, 0]
model = load_model("model.h5")
preY_vec = []
# 用生成的参数按时间跑一边
for i in range(1, n-5):
    ord_x = x_data[i] # 此处应调整。取出来的不知道是啥
    preY = model.predict(ord_x)
    preY_vec.append(preY)

fig = plt.figure()
plt.plot(y_data, 'g-', label="Exact", linewidth=0.2)
plt.plot(preY_vec, 'r-', label="Prediction", linewidth=0.4)
plt.title('Rate @ Time')
plt.xlabel('Time')
plt.ylabel('Rate')