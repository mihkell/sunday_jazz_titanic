import os

import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import LabelBinarizer

pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 100000)
resource_path = '../../data/'
file_name = 'picled_data.pickle'


def normalize(df):

    AGE = 'Age'
    ageMax = df[AGE].max()
    ageMin = df[AGE].min()
    df[AGE] = (df[AGE]-ageMin)/(ageMax-ageMin)

    FARE = 'Fare'
    fareMax = df[FARE].max()
    fareMin = df[FARE].min()
    df[FARE] = (df[FARE]-fareMin)/(fareMax-fareMin)

    return df

def clean_data(data_filename):
    file_path = resource_path + file_name
    if not os.path.exists(file_path):
        df = pd.read_csv('../../data/' + data_filename)
        SEX = 'Sex'
        df = df[['Survived', 'Pclass', 'Age', 'Fare', SEX]]
        df = df.dropna()

        df[SEX] = LabelBinarizer().fit_transform(df[SEX])
        df['Sex2'] = (df[SEX] - 1) * -1

        df = normalize(df)

        pd.to_pickle(df, file_path)
    return pd.read_pickle(file_path)


def get_features_labels(data):
    return data[['Pclass', 'Age', 'Fare', 'Sex', 'Sex2']].as_matrix(), data[['Survived']].as_matrix()

def learn(df):
    # Model parameters
    W = tf.Variable(tf.ones([5, 1]), tf.float32)
    b = tf.Variable(tf.ones(1), tf.float32)
    # Model input and output
    x = tf.placeholder(tf.float32, shape=(None, 5))
    linear_model = tf.matmul(x, W) + b
    y = tf.placeholder(tf.float32, shape=(None, 1))

    loss = tf.reduce_sum(tf.abs(linear_model-y))
    # optimizer
    optimizer = tf.train.GradientDescentOptimizer(0.01)
    train = optimizer.minimize(loss)

    # Our data example
    x_train, y_train = get_features_labels(df)
    # training loop
    init = tf.global_variables_initializer()
    sess = tf.Session()
    sess.run(init)
    range_val = 600
    batch_size = 5
    for i in range(range_val):
        for batch_start in range(0, len(x_train) - batch_size, batch_size):
            batch_end = batch_start + batch_size
            sess.run(train, {x: x_train[batch_start:batch_end], y: y_train[batch_start:batch_end]})
            # if range_val%100 == 0:
            #     curr_W, curr_b, curr_loss = sess.run([W, b, loss], {x: x_train, y: y_train})
            #     print("W: %s b: %s loss: %s" % (curr_W, curr_b, curr_loss))

    curr_W, curr_b, curr_loss = sess.run([W, b, loss], {x: x_train, y: y_train})
    print("W: %s b: %s loss: %s" % (curr_W, curr_b, curr_loss))
    # evaluate training accuracy


def main():
    df = clean_data("train.csv")
    learn(df)
    # print(df[['Survived']].as_matrix().transpose())
    # i = 0
    # for row in df.iterrows():
    #     print(row)
    #     i += 1
    #     if i == 4:
    #         break


if __name__ == "__main__":
    main()
