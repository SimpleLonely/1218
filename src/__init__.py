# coding=utf-8
import tensorflow as tf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
from tensorflow.python import debug as tf_debug

LEARNING_RATE_BASE = 1e-3
LEARNING_RATE_DECAY = 0.98
LEARNING_RATE_STEP = 300

def normalize_cols(m):
    col_max = m.max(axis=0)
    col_min = m.min(axis=0)
    return (m-col_min)/(col_max-col_min)


sess = tf.InteractiveSession()


sess = tf_debug.LocalCLIDebugWrapperSession(sess)

sess.add_tensor_filter("has_inf_or_nan", tf_debug.has_inf_or_nan)

if __name__ == "__main__":


#def run_one_time(params):

#    LEARNING_RATE_DECAY = params["LEARNING_RATE_DECAY"]
#    LEARNING_RATE_STEP = params["LEARNING_RATE_STEP"]
#    n_neurons_1 = params["n_neurons_1"]
#    n_neurons_2 = params["n_neurons_2"]
#    n_neurons_3 = params["n_neurons_3"]
#    batch_size = params["batch_size"]
#    round = params["round"]

    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

    data = pd.read_csv('../res/input0420.csv')
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

    n_parameters = 5

    n_target = 1

    X = tf.placeholder(dtype=tf.float32, shape=[None, n_parameters], name='x-input')
    Y = tf.placeholder(dtype=tf.float32, shape=[None, 1], name='y-input')


    W_hidden_1 = tf.Variable(tf.truncated_normal([n_parameters, n_neurons_1], stddev=1, mean=0))
    bias_hidden_1 = tf.Variable(tf.truncated_normal([n_neurons_1], mean=0))
    hidden_1 = tf.nn.softsign(tf.add(tf.matmul(X, W_hidden_1), bias_hidden_1))

    W_hidden_2 = tf.Variable(tf.truncated_normal([n_neurons_1, n_neurons_2], stddev=1, mean=0))
    bias_hidden_2 = tf.Variable(tf.truncated_normal([n_neurons_2], mean=0))
    hidden_2 = tf.nn.tanh(tf.add(tf.matmul(hidden_1, W_hidden_2), bias_hidden_2))

    W_hidden_3 = tf.Variable(tf.truncated_normal([n_neurons_2, n_neurons_3], stddev=1, mean=0))
    bias_hidden_3 = tf.Variable(tf.truncated_normal([n_neurons_3], mean=0))

    hidden_3 = tf.nn.softsign(tf.add(tf.matmul(hidden_2, W_hidden_3), bias_hidden_3))

    W_out = tf.Variable(tf.truncated_normal([n_neurons_3, n_target], stddev=1, mean=0))
    bias_out = tf.Variable(tf.truncated_normal([n_target], mean=0))
    out = tf.transpose(tf.add(tf.matmul(hidden_3, W_out), bias_out))

    global_step = tf.Variable(0, trainable=False)

    loss = tf.reduce_mean(np.square(out-Y))

    learning_rate = tf.train.exponential_decay(LEARNING_RATE_BASE, global_step, LEARNING_RATE_STEP, LEARNING_RATE_DECAY, staircase=True)
    my_opt = tf.train.GradientDescentOptimizer(learning_rate)
    train_step = my_opt.minimize(loss, global_step=global_step)

    init = tf.global_variables_initializer()
    sess.run(init)
    saver = tf.train.Saver({"w1":W_hidden_1,"w2":W_hidden_2,"w3":W_hidden_3})
    loss_vec = []
    test_loss = []
    pre_y_vec = []
    exactY_vec = []

    batch_size = 20
    step = 2000
    last_lost = 0

    for i in range(step):
        rand_index = np.random.choice(len(x_train), size=batch_size)
        rand_x = x_train[rand_index]
        rand_y = np.transpose([y_train[rand_index]])

        sess.run(train_step, feed_dict={X: rand_x, Y: rand_y})

        learning_rate_val = sess.run(learning_rate)
        global_step_val = sess.run(global_step)

        #temp_w1 = sess.run(W_hidden_1, feed_dict={X: rand_x, Y: rand_y})
        #temp_b1 = sess.run(bias_hidden_1, feed_dict={X: rand_x, Y: rand_y})
        #temp_h1 = sess.run(hidden_1, feed_dict={X: rand_x, Y: rand_y})
        #temp_h2 = sess.run(hidden_2, feed_dict={X: rand_x, Y: rand_y})
        #temp_h3 = sess.run(hidden_3, feed_dict={X: rand_x, Y: rand_y})
        #temp_out = sess.run(out, feed_dict={X: rand_x, Y: rand_y})
        # temp_x = sess.run(X, feed_dict={X: rand_x, Y: rand_y})
        #temp_y = sess.run(Y, feed_dict={X: rand_x, Y: rand_y})
        temp_loss = sess.run(loss, feed_dict={X: rand_x, Y: rand_y})
        loss_vec.append(np.sqrt(temp_loss))

        test_temp_loss = sess.run(loss, feed_dict={X: x_test, Y: np.transpose([y_test])})
        test_loss.append(np.sqrt(test_temp_loss))

        if (last_lost - temp_loss) * (last_lost - temp_loss) <= 1:
            print("It's the end!")

        last_lost = temp_loss

        if(i+1) % 5 == 0:
            print("***********************")
            print("%s steps:rate is %s" % (global_step_val,learning_rate_val))
            print('Generation' + str(i+1) + '.Loss = ' + str(temp_loss))


        if np.isnan(temp_loss):
            print("When NAN showed up: ")
            #print("X:" + str(temp_x))
            #print("W_Hidden_1:" + str(temp_w1))
            #print("Bias_Hidden_1:" + str(temp_b1))
            #print("Hidden_1:" + str(temp_h1))
            #print("Y:"+str(temp_y))
            #print("out:"+str(temp_out))
            exit(0)

        #if (i+1)%100 == 0:
        #    saver.save(sess,"../result/"+str(n_neurons_1)+"_"+str(n_neurons_2)+"_"+str(n_neurons_3)+"params",global_step=i)

    # 用生成的参数按时间跑一边
    for i in range(1, n-5):
        ord_index = np.random.random_integers(i, i+1, 2)
        ord_x = x_data[ord_index]
        ord_y = np.transpose([y_data[ord_index]])

        # sess.run(train_step, feed_dict={X: ord_x, Y: ord_y})

        exactY = sess.run(Y, feed_dict={X: ord_x, Y: ord_y})
        preY = sess.run(out, feed_dict={X: ord_x, Y: ord_y})

        pre_y_vec.append(preY[0][0])
        exactY_vec.append(exactY[0][0])

    fig = plt.figure()
    plt.plot(exactY_vec, 'g-', label="Exact", linewidth=0.2)
    plt.plot(pre_y_vec, 'r-', label="Prediction", linewidth=0.4)
    plt.title('Rate @ Time')
    plt.xlabel('Time')
    plt.ylabel('Rate')
    fig.savefig("../img/"+str('_'.join([str(values) for key,values in params.items()]))+"_Prediction.png")
    plt.show()

    fig = plt.figure()
    plt.plot(loss_vec, 'k-', label='Train Loss')
    plt.plot(test_loss, 'r--', label='Test Loss')
    plt.title('Loss per Generation')
    plt.xlabel('Generation')
    plt.ylabel('Loss')
    plt.legend(loc='upper right')

    fig.savefig("../img/loss/"+str('_'.join([str(values) for key,values in params.items()]))+"_Loss.png")
    plt.show()


if __name__ == "__main__":
    for learning_decay in range(5, 9):
        for learning_step in range(50, 1000, 10):
            for i in range(4,129):
                for j in range(1,i):
                    for k in range(1,j):
                        for rounds in range(50,50000,100):
                            for batch_size in range(5,100):
                                params = {"LEARNING_RATE_DECAY":float(learning_decay)/10,"LEARNING_RATE_STEP":learning_step,"n_neurons_1":i,"n_neurons_2":j,"n_neurons_3":k,"batch_size":batch_size,"round":rounds}
                                run_one_time(params)