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
exacY_vec = []
# ord_x = x_data.iloc[[0],:]
# preY = model.predict(ord_x)
# print(float(preY[0]))
# 用生成的参数按时间跑一边
for i in range(0, n-5):
    ord_x = x_data.iloc[[i],:]
    preY = model.predict(ord_x)
    preY_vec.append(float(preY[0]))
    exacY_vec.append(y_data.iloc[0])
print (preY_vec)
fig = plt.figure()
plt.plot(exacY_vec, 'g-', label="Exact", linewidth=0.2)
plt.plot(preY_vec, 'r-', label="Prediction", linewidth=0.4)
plt.title('Rate @ Time')
plt.xlabel('Time')
plt.ylabel('Rate')
plt.show()