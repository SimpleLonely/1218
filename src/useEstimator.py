# coding=utf-8
import tensorflow as tf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
from tensorflow.python import debug as tf_debug

PATH = "/estimator/"
PATH_DATASET = PATH + "res"
FILE_TRAIN = PATH_DATASET + os.sep + "training.csv"
FILE_TEST = PATH_DATASET + os.sep + "iris_test.csv"

feature_names = ["BIGpf","LGdija","usdin","loggas","logauf"]

def my_input_func(file_path,perform_shuffle=False,repeat_count=1):
    def decode_csv(line):
        parsed_line = tf.decode_csv(line, [[0.],[""], [0.], [0.], [0.],[0.],[0.]])
        label = parsed_line[0]  # First element is the label
        del parsed_line[0]  # Delete first element
        del parsed_line[0]  # Del Date
        features = parsed_line  # Everything but first elements are the features
        d = dict(zip(feature_names, features)), label
        return d
    dataset = (tf.data.TextLineDataset(file_path)  # Read text file 读取文本
               .skip(1)  # Skip header row # 跳过第一行表头
               .map(decode_csv))  # Transform each elem by applying decode_csv fn # 将数据进行转换
    print(dataset)
    if perform_shuffle:
        # Randomizes input using a window of 256 elements (read into memory)
        dataset = dataset.shuffle(buffer_size=256)
    dataset = dataset.repeat(repeat_count)  # Repeats dataset this # times
    dataset = dataset.batch(32)  # Batch size to use
    iterator = dataset.make_one_shot_iterator()
    batch_features, batch_labels = iterator.get_next()
    return batch_features, batch_labels


next_batch = my_input_func("../res/input0420.csv", True)

# Create the feature_columns, which specifies the input to our model
# All our input features are numeric, so use numeric_column for each one
feature_column = [tf.feature_column.numeric_column(k) for k in feature_names]

# Create a deep neural network regression classifier
# Use the DNNClassifier pre-made estimator
regressor = tf.estimator.DNNRegressor(
    feature_columns=feature_column,  # The input features to our model
    hidden_units=[10, 10],  # Two layers, each with 10 neurons
    model_dir=PATH)  # Path to where checkpoints etc are stored
evaluations = []  
STEPS = 400
print("*********Start Training***********")
regressor.train(input_fn=lambda:my_input_func(PATH,True))
evaluations.append(regressor.evaluate(input_fn=lambda:my_input_func(PATH,False)))
print("***********Evaluation results:")
for key in evaluations:
    print("   {}, was: {}".format(key, evaluations[key]))