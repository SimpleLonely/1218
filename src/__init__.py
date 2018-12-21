# coding=utf-8
import tensorflow as tf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

sess = tf.Session()

data = pd.read_csv('E:/1218/res/data_meg.csv')
n = data.shape[0]
m = data.shape[1]
train_start = 0
train_end = int(np.floor(0.8*n))
test_start = train_end + 1
test_end = n
data_train = data.loc[train_start: train_end]
data_test = data.loc[test_start: test_end]

x_train = data_train.ix[:, 1:]
y_train = data_train.ix[:, 0]
x_test = data_test.ix[:, 1:]
y_test = data_test.ix[:, 0]


def normalize_cols(m):
    col_max = m.max(axis=0)
    col_min = m.min(axis=0)
    return (m-col_min)/(col_max-col_min)


x_test = np.nan_to_num(normalize_cols(x_test))
x_train = np.nan_to_num(normalize_cols(x_train))

print(x_test)

n_parameters = 3
n_neurons_1 = 64
n_neurons_2 = 16
n_neurons_3 = 4
n_target = 1

X = tf.placeholder(dtype=tf.float32, shape=[None, n_parameters], name='x-input')
Y = tf.placeholder(dtype=tf.float32, shape=[None, 1], name='y-input')

W_hidden_1 = tf.Variable(tf.random_normal([n_parameters, n_neurons_1]))
bias_hidden_1 = tf.Variable(tf.random_normal([n_neurons_1]))
hidden_1 = tf.nn.relu(tf.add(tf.matmul(X, W_hidden_1), bias_hidden_1))

W_hidden_2 = tf.Variable(tf.random_normal([n_neurons_1, n_neurons_2]))
bias_hidden_2 = tf.Variable(tf.random_normal([n_neurons_2]))
hidden_2 = tf.nn.relu(tf.add(tf.matmul(hidden_1, W_hidden_2), bias_hidden_2))

W_hidden_3 = tf.Variable(tf.random_normal([n_neurons_2, n_neurons_3]))
bias_hidden_3 = tf.Variable(tf.random_normal([n_neurons_3]))
hidden_3 = tf.nn.relu(tf.add(tf.matmul(hidden_2, W_hidden_3), bias_hidden_3))

W_out = tf.Variable(tf.random_normal([n_neurons_3, n_target]))
bias_out = tf.Variable(tf.random_normal([n_target]))
out = tf.transpose(tf.add(tf.matmul(hidden_3, W_out), bias_out))

loss = tf.reduce_mean(tf.square(out-Y))

my_opt = tf.train.GradientDescentOptimizer(0.005)
train_step = my_opt.minimize(loss)

init = tf.global_variables_initializer()
sess.run(init)

loss_vec = []
test_loss = []
batch_size = 20

for i in range(5000):
    rand_index = np.random.choice(len(x_train), size=batch_size)
    rand_x = x_train[rand_index]
    rand_y = np.transpose([y_train[rand_index]])
    sess.run(train_step, feed_dict={X: rand_x, Y: rand_y})

    temp_loss = sess.run(loss, feed_dict={X: rand_x, Y: rand_y})
    loss_vec.append(np.sqrt(temp_loss))

    test_temp_loss = sess.run(loss, feed_dict={X: x_test, Y: np.transpose([y_test])})
    test_loss.append(np.sqrt(test_temp_loss))

    if(i+1)%50 == 0:
        print('Generation' + str(i+1) + '.Loss = ' + str(temp_loss))


plt.plot(loss_vec, 'k-', label='Train Loss')
plt.plot(test_loss, 'r--', label='Test Loss')
plt.title('Loss per Generation')
plt.xlabel('Generation')
plt.ylabel('Loss')
plt.legend(loc='upper right')
plt.show()
