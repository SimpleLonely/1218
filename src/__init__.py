# coding=utf-8
import tensorflow as tf
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

BATCH_SIZE = 200
LEARNING_RATE_BASE = 0.1
LEARNING_RATE_DECAY = 0.99
REGULARIZER = 0.0001
STEPS=50000
MOVING_AVERAGE_DECAY = 0.99
MODEL_SAVE_PATH = "./model/"
MODEL_NAME = "1218_model"

sess = tf.Session()

data = pd.read_csv('E:/1218/res/data_meg.csv')
n = data.shape[0]
m = data.shape[1]

train_start = 0
train_end = int(np.floor(0.8*n))
test_start = train_end + 1
test_end = n
data_train = data[np.arange(train_start, train_end), :]
data_test = data[np.arange(test_start, test_end), :]

scaler = MinMaxScaler()
scaler.fit(data_train)
scaler.transform(data_train)
scaler.transform(data_test)

X_train = data_train[:, 1:]
y_train = data_train[:, 0]
X_test = data_test[:, 1:]
y_test = data_test[:, 0]

n_parameters = 3
n_neurons_1 = 256
n_neurons_2 = 64
n_neurons_3 = 16
n_target = 1

weight_initializer = tf.variance_scaling_initializer(mode="fan_avg", distribution="uniform")
bias_initializer = tf.zeros_initializer()


def train_rnn():
    X = tf.placeholder(dtype=tf.float32, shape=[None, n_parameters], name='x-input')
    Y = tf.placeholder(dtype=tf.float32, shape=[None], name='y-input')

    W_hidden_1 = tf.Variable(weight_initializer([n_parameters, n_neurons_1]))
    bias_hidden_1 = tf.Variable(bias_initializer([n_neurons_1]))
    hidden_1 = tf.nn.relu(tf.add(tf.matmul(X, W_hidden_1), bias_hidden_1))

    W_hidden_2 = tf.Variable(weight_initializer([n_neurons_1, n_neurons_2]))
    bias_hidden_2 = tf.Variable(bias_initializer([n_neurons_2]))
    hidden_2 = tf.nn.relu(tf.add(tf.matmul(hidden_1, W_hidden_2), bias_hidden_2))

    W_hidden_3 = tf.Variable(weight_initializer([n_neurons_2, n_neurons_3]))
    bias_hidden_3 = tf.Variable(bias_initializer([n_neurons_3]))
    hidden_3 = tf.nn.relu(tf.add(tf.matmul(hidden_2, W_hidden_3), bias_hidden_3))

    W_out = tf.Variable(weight_initializer([n_neurons_3, n_target]))
    bias_out = tf.Variable(bias_initializer([n_target]))
    out = tf.transpose(tf.add(tf.matmul(hidden_3, W_out), bias_out))

    mse = tf.reduce_mean(tf.squared_difference(out, Y))

    global_step = tf.Variable(0, trainable=False)
    learning_rate=tf.train.exponential_decay(
        LEARNING_RATE_BASE,
        global_step,
        (train_end-train_start+1)/BATCH_SIZE,
        LEARNING_RATE_DECAY,
        staircase=True
    )

    train_step = tf.train.GradientDescentOptimizer(learning_rate).minimize(mse,global_step=global_step)

    ema = tf.train.ExponentialMovingAverage(MOVING_AVERAGE_DECAY, global_step)
    ema_op = ema.apply(tf.trainable_variables())
    with tf.control_dependencies([train_step, ema_op]):
        train_op = tf.no_op(name='train')

    saver = tf.train.Saver()

    with tf.Session() as sess:
        init_op = tf.global_variables_initializer()
        sess.run(init_op)

        for i in range(STEPS):
            xs,ys=data_train.train.next_batch(BATCH_SIZE)
            _, loss_value, step = sess.run([train_op, mse, global_step], feed_dict={X:xs, Y:ys})
            if i% 1000 == 0:
                print("After %d train step(s), loss on training batch is %g." % (step, loss_value))
                saver.save(sess.os.path.join(MODEL_SAVE_PATH,MODEL_NAME),global_step=global_step)

    # Setup interactive plot
    plt.ion()
    fig = plt.figure()
    ax1 = fig.add_subplot(111)
    line1, = ax1.plot(y_test)
    line2, = ax1.plot(y_test * 0.5)
    plt.show()

train_rnn()