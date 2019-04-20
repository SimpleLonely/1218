# coding=utf-8
import tensorflow as tf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os

LEARNING_RATE_BASE = 0.1
LEARNING_RATE_DECAY = 0.5
LEARNING_RATE_STEP = 50


def normalize_cols(m):
    col_max = m.max(axis=0)
    col_min = m.min(axis=0)
    return (m-col_min)/(col_max-col_min)


sess = tf.InteractiveSession()

if __name__ == "__main__":
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

    # import data
    data = pd.read_csv('E:/1218/res/input0420.csv')
    o_file = open("E:/1218/res/out_put.txt", "w")
    n = data.shape[0]
    m = data.shape[1]
    train_start = 1
    train_end = int(np.floor(0.8 * n))
    test_start = train_end + 1
    test_end = n
    data_train = data.loc[train_start: train_end]
    data_test = data.loc[test_start: test_end]

    x_train = data_train.ix[:, 2:]
    y_train = data_train.ix[:, 0]
    x_test = data_test.ix[:, 2:]
    y_test = data_test.ix[:, 0]
    x_test = np.nan_to_num(normalize_cols(x_test))
    x_train = np.nan_to_num(normalize_cols(x_train))

    n_parameters = 5
    n_neurons_1 = 64
    n_neurons_2 = 16
    n_neurons_3 = 4
    n_target = 1

    X = tf.placeholder(dtype=tf.float32, shape=[None, n_parameters], name='x-input')
    Y = tf.placeholder(dtype=tf.float32, shape=[None, 1], name='y-input')

    W_hidden_1 = tf.Variable(tf.truncated_normal([n_parameters, n_neurons_1], stddev=1, mean=0))
    bias_hidden_1 = tf.Variable(tf.truncated_normal([n_neurons_1], mean=0))
    hidden_1 = tf.nn.relu(tf.add(tf.matmul(X, W_hidden_1), bias_hidden_1))

    W_hidden_2 = tf.Variable(tf.truncated_normal([n_neurons_1, n_neurons_2], stddev=1, mean=0))
    bias_hidden_2 = tf.Variable(tf.truncated_normal([n_neurons_2], mean=0))
    hidden_2 = tf.nn.relu(tf.add(tf.matmul(hidden_1, W_hidden_2), bias_hidden_2))

    W_hidden_3 = tf.Variable(tf.truncated_normal([n_neurons_2, n_neurons_3], stddev=1, mean=0))
    bias_hidden_3 = tf.Variable(tf.truncated_normal([n_neurons_3], mean=0))
    hidden_3 = tf.nn.relu(tf.add(tf.matmul(hidden_2, W_hidden_3), bias_hidden_3))

    W_out = tf.Variable(tf.truncated_normal([n_neurons_3, n_target], stddev=1, mean=0))
    bias_out = tf.Variable(tf.truncated_normal([n_target], mean=0))
    out = tf.transpose(tf.add(tf.matmul(hidden_3, W_out), bias_out))

    global_step = tf.Variable(0, trainable=False)

    loss = tf.reduce_mean(tf.square(out-Y))

    learning_rate = tf.train.exponential_decay(LEARNING_RATE_BASE, global_step, LEARNING_RATE_STEP, LEARNING_RATE_DECAY, staircase=True)
    my_opt = tf.train.GradientDescentOptimizer(learning_rate)
    train_step = my_opt.minimize(loss, global_step=global_step)

    init = tf.global_variables_initializer()
    sess.run(init)

    loss_vec = []
    test_loss = []
    batch_size = 20

    for i in range(500):
        rand_index = np.random.choice(len(x_train), size=batch_size)
        rand_x = x_train[rand_index]
        rand_y = np.transpose([y_train[rand_index]])
        sess.run(train_step, feed_dict={X: rand_x, Y: rand_y})

        learning_rate_val = sess.run(learning_rate)
        global_step_val = sess.run(global_step)

        temp_loss = sess.run(loss, feed_dict={X: rand_x, Y: rand_y})
        loss_vec.append(np.sqrt(temp_loss))

        test_temp_loss = sess.run(loss, feed_dict={X: x_test, Y: np.transpose([y_test])})
        test_loss.append(np.sqrt(test_temp_loss))

        # print(out.eval())
        # tf.Print(out, [out])
        # print("&"),
        # tf.Print(Y, [Y]),
        # print("&"),
        # tf.Print(loss, [loss]),
        # print("&")
        # print()

        if(i + 1) % 100 == 0:
            print("%s steps:rate is %s" % (global_step_val, learning_rate_val))
            print('Generation' + str(i+1) + '.Loss = ' + str(temp_loss))

    print("loss_vec %d:", len(loss_vec), file=o_file)
    for v in loss_vec:
        print(v, file=o_file)
    print("test_loss %d:", len(test_loss), file=o_file)
    for v in test_loss:
        print(v, file=o_file)

    o_file.close()

    plt.plot(loss_vec, 'k-', label='Train Loss')
    plt.plot(test_loss, 'r--', label='Test Loss')

    plt.title('Loss per Generation')
    plt.xlabel('Generation')
    plt.ylabel('Loss')
    plt.legend(loc='upper right')
    plt.show()

