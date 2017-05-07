import os

import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelBinarizer

pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 100000)
resource_path = '../../data/'
file_name = 'picled_data.pickle'


def normalize(df):
    AGE = 'Age'
    ageMax = df[AGE].max()
    ageMin = df[AGE].min()
    df[AGE] = (df[AGE] - ageMin) / (ageMax - ageMin)
    df[AGE] = (df[AGE]**0.8 * 2 - 1)

    FARE = 'Fare'
    fareMax = df[FARE].max()
    fareMin = df[FARE].min()
    df[FARE] = (df[FARE] - fareMin) / (fareMax - fareMin)

    return df


def clean_data(data_filename):
    file_path = resource_path + file_name
    # if not os.path.exists(file_path):
    df = pd.read_csv('../../data/' + data_filename)
    SEX = 'Sex'
    AGE = 'Age'
    df = df[['Survived', 'Pclass', AGE, 'Fare', SEX]]
    age_missing = 'AgeMissing'
    df[age_missing] = df[AGE].isnull()
    df[age_missing] = LabelBinarizer().fit_transform(df[age_missing]) * 2 - 1

    df = df.fillna(method='bfill')
    df = df.fillna(method='ffill')

    df[SEX] = LabelBinarizer().fit_transform(df[SEX])
    df[SEX] = df[SEX] * 2 - 1


    df = normalize(df)

    pd.to_pickle(df, file_path)
    return pd.read_pickle(file_path)


def get_features_labels(data):
    return data[['Sex']].as_matrix(), data[['Survived']].as_matrix()


def learn(df, test_set):
    # Try next time:
    # http://stackoverflow.com/questions/33919948/how-to-set-adaptive-learning-rate-for-gradientdescentoptimizer

    x_train, y_train = get_features_labels(df)
    x_test, y_test = get_features_labels(test_set)



    inputs = tf.placeholder(tf.float32, shape=(None, x_train.shape[1]))
    labels = tf.placeholder(tf.float32, shape=(None, 1))
    # make forward pass of layer
    # layer_sigmoid = tf.contrib.layers.fully_connected(inputs, 1,
    #                                                   activation_fn=tf.nn.softplus,
    #                                                   weights_initializer=tf.zeros_initializer())

    # layer_sigmoid = tf.contrib.layers.fully_connected(inputs, 2,
    #                                                   activation_fn=tf.nn.softplus,
    #                                                   weights_initializer=tf.zeros_initializer())

    # layer_sigmoid = tf.contrib.layers.fully_connected(inputs, 3,
    #                                                   activation_fn=tf.nn.softplus,
    #                                                   weights_initializer=tf.zeros_initializer())

    logits = tf.contrib.layers.fully_connected(inputs, 1,
                                               activation_fn=tf.sigmoid,
                                               weights_initializer=tf.zeros_initializer())
    # accuracy
    accuracy = tf.abs(logits - labels)
    accuracy = tf.count_nonzero(accuracy < 0.5) / tf.placeholder(tf.int64, name='input_size')

    # training
    loss = tf.reduce_sum(tf.abs(logits - labels))
    optimizer = tf.train.GradientDescentOptimizer(0.01)
    train = optimizer.minimize(loss)

    # running graph
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        batch_size = 10
        for i in range(50):  # 2 rounds of training
            for batch_start in range(0, len(x_train) - batch_size, batch_size):
                batch_end = batch_start + batch_size
                features = x_train[batch_start:batch_end]
                labels_ = y_train[batch_start:batch_end]
                sess.run(train, {inputs: features,
                                 labels: labels_})

            sigmoid_layer_val, ran_loss_val, accuracy_val = sess.run([inputs, loss, accuracy],
                                                         {inputs: x_test, labels: y_test,
                                                          'input_size:0': y_test.shape[0]})
            print(i, 'accuracy/loss:', accuracy_val, ran_loss_val)


def main():
    data = clean_data("train.csv")
    print('data-size-total:', data.shape[0])
    print(data[data['Survived'] == 1].shape[0])
    train_set, test_test = train_test_split(data, random_state=1)
    learn(train_set, test_test)
    # print(df[['Survived']].as_matrix().transpose())
    # i = 0
    # for row in df.iterrows():
    #     print(row)
    #     i += 1
    #     if i == 4:
    #         break


if __name__ == "__main__":
    main()
