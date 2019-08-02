# coding=utf-8
import tensorflow as tf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

LEARNING_RATE_BASE = 1e-2
LEARNING_RATE_DECAY = 0.98
LEARNING_RATE_STEP = 100

sess = tf.Session()


def normalize_cols_a(m0):
    m1 = m0.copy()
    for it in m1.index:
        if -14.5 <= m1[it] <= -2.1:
            m1[it] = (m1[it] + 14.5) / 12.4 / 16
        elif -2.1 <= m1[it] <= -1.4:
            m1[it] = (m1[it] + 2.1) / 0.7 / 16 + 0.0625
        elif -1.4 <= m1[it] <= -0.7:
            m1[it] = (m1[it] + 1.4) / 0.7 / 8 + 0.125
        elif -0.7 <= m1[it] <= 0:
            m1[it] = (m1[it] + 0.7) / 0.7 / 4 + 0.25
        elif 0 <= m1[it] <= 0.7:
            m1[it] = m1[it] / 0.7 / 4 + 0.5
        elif 0.7 <= m1[it] <= 1.4:
            m1[it] = (m1[it] - 0.7) / 0.7 / 8 + 0.75
        elif 1.4 <= m1[it] <= 2.1:
            m1[it] = (m1[it] - 1.4) / 0.7 / 16 + 0.875
        elif 2.1 <= m1[it] <= 14.5:
            m1[it] = (m1[it] - 2.1) / 12.4 / 16 + 0.9375

    return m1


def reverse_normalize_cols_a(m0):
    m1 = m0.copy()
    for it in range(len(m1)):
        if 0 <= m1[it] <= 0.0625:
            m1[it] = m1[it] * 12.4 * 16 - 14.5
        elif 0.0625 <= m1[it] <= 0.125:
            m1[it] = (m1[it] - 0.0625) * 0.7 * 16 - 2.1
        elif 0.125 <= m1[it] <= 0.25:
            m1[it] = (m1[it] - 0.125) * 0.7 * 8 - 1.4
        elif 0.25 <= m1[it] <= 0.5:
            m1[it] = (m1[it] - 0.25) * 0.7 * 4 - 0.7
        elif 0.5 <= m1[it] <= 0.75:
            m1[it] = (m1[it] - 0.5) * 0.7 * 4
        elif 0.75 <= m1[it] <= 0.875:
            m1[it] = (m1[it] - 0.75) * 0.7 * 8 + 0.7
        elif 0.875 <= m1[it] <= 0.9375:
            m1[it] = (m1[it] - 0.875) * 0.7 * 16 + 1.4
        elif 0.09375 <= m1[it] <= 1:
            m1[it] = (m1[it] - 0.9375) * 12.4 * 16 + 2.1

    return m1


def normalize_cols_b(m1):
    col_min = m1.min(axis=0)
    col_max = m1.max(axis=0)
    return (m1 - col_min) / (col_max - col_min)


def init_variable(shape):
    return tf.Variable(tf.truncated_normal(shape, mean=0, stddev=1))


def logistic(input_layer, multiplication_weight, bias_weight, activation=True):
    layer = tf.add(tf.matmul(input_layer, multiplication_weight), bias_weight)
    if activation:
        return tf.nn.softsign(layer)
    else:
        return layer


# import data
data = pd.read_csv('../res/input0701.csv')
n = data.shape[0]
m = data.shape[1]
train_start = 1
train_end = int(np.floor(0.8 * n))
test_start = train_end + 1
test_end = n

data_train = data.loc[train_start: train_end]
data_test = data.loc[test_start: test_end]

y_train = np.nan_to_num(normalize_cols_a(data_train.ix[:, 0]))
y_test = np.nan_to_num(normalize_cols_a(data_test.ix[:, 0]))
x_test = np.nan_to_num(normalize_cols_b(data_test.ix[:, 2:]))
x_train = np.nan_to_num(normalize_cols_b(data_train.ix[:, 2:]))

n_parameters = 31
n_neurons_1 = 100
n_neurons_2 = 50
n_neurons_3 = 3
n_target = 1

X = tf.placeholder(dtype=tf.float32, shape=[None, n_parameters], name='x-input')
Y = tf.placeholder(dtype=tf.float32, shape=[None, 1], name='y-input')

A1 = init_variable([n_parameters, n_neurons_1])
B1 = init_variable([n_neurons_1])
L1 = logistic(X, A1, B1)

A2 = init_variable([n_neurons_1, n_neurons_2])
B2 = init_variable([n_neurons_2])
L2 = logistic(L1, A2, B2)

A3 = init_variable([n_neurons_2, n_neurons_3])
B3 = init_variable([n_neurons_3])
L3 = logistic(L2, A3, B3)

A0 = init_variable([n_neurons_3, n_target])
B0 = init_variable([n_target])
out = logistic(L3, A0, B0)

global_step = tf.Variable(0, trainable=False)

# loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=out, logits=Y))
loss = tf.reduce_mean(tf.squared_difference(out, Y))

learning_rate = tf.train.exponential_decay(LEARNING_RATE_BASE, global_step, LEARNING_RATE_STEP, LEARNING_RATE_DECAY,
                                           staircase=True)
my_opt = tf.train.GradientDescentOptimizer(learning_rate)
train_step = my_opt.minimize(loss, global_step=global_step)

init = tf.global_variables_initializer()
sess.run(init)

loss_vec = []

batch_size = 20
step = 500

for i in range(step):
    rand_index = np.random.choice(np.arange(1, len(x_train)), size=batch_size)
    rand_x = x_train[rand_index]
    rand_y = np.transpose([y_train[rand_index]])

    sess.run(train_step, feed_dict={X: rand_x, Y: rand_y})

    temp_loss = sess.run(loss, feed_dict={X: rand_x, Y: rand_y})
    loss_vec.append(temp_loss)

    if (i + 1) % 5 == 0:
        print('Gen ' + str(i+1) + ' Loss = ' + str(temp_loss))

plt.plot(loss_vec, 'k-')
plt.title('Loss per Generation')
plt.xlabel('Generation')
plt.ylabel('Loss')
plt.show()

exactY_train = data_train.ix[:, 0]
preY0_train = sess.run(out, feed_dict={X: x_train, Y: np.transpose([y_train])})
preY1_train = reverse_normalize_cols_a(preY0_train)

plt.plot(exactY_train, 'g-', label="Exact", linewidth=0.2)
plt.plot(preY1_train, 'r-', label="Prediction", linewidth=0.4)
plt.title('Rate_train @ Time')
plt.xlabel('Time')
plt.ylabel('Rate_train')
plt.show()

exactY_test = data_test.ix[:, 0]
preY0_test = sess.run(out, feed_dict={X: x_test, Y: np.transpose([y_test])})
preY1_test = reverse_normalize_cols_a(preY0_test)

plt.plot(exactY_test, 'g-', label="Exact", linewidth=0.2)
plt.plot(preY1_test, 'r-', label="Prediction", linewidth=0.4)
plt.title('Rate_test @ Time')
plt.xlabel('Time')
plt.ylabel('Rate_test')
plt.show()
