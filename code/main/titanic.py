import os

import pandas as pd
import tensorflow as tf
from sklearn.preprocessing import LabelBinarizer

pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 100000)
resource_path = '../../data/'
file_name = 'picled_data.pickle'


def clean_data(data_filename):
    file_path = resource_path + file_name
    if not os.path.exists(file_path):
        df = pd.read_csv('../../data/' + data_filename)
        SEX = 'Sex'
        df = df[['Survived', 'Pclass', 'Age', 'Fare', SEX]]
        df = df.dropna()
        df[SEX] = LabelBinarizer().fit_transform(df[SEX])
        df['Sex2'] = (df[SEX] - 1) * -1
        pd.to_pickle(df, file_path)
    return pd.read_pickle(file_path)


def get_features_labels(data):
    return data[['Pclass', 'Age', 'Fare', 'Sex', 'Sex2']].as_matrix(), data[['Survived']].as_matrix()


def learn(data):
    features, labels = get_features_labels(data)
    print(features[:3], labels[:3])
    with tf.name_scope('Variable_declaration'):
        W = tf.Variable(tf.ones([5, 1]), name="Weight")
        b = tf.Variable([-.3], name="Bias")

    input_ = tf.placeholder(tf.float32, shape=(None, 5), name="input")
    target = tf.placeholder(tf.float32, name="target")

    with tf.name_scope('linear_model'):
        linear_model = tf.matmul(input_, W) + b

    with tf.name_scope('optimizing'):
        loss = tf.abs(tf.reduce_sum(linear_model - target))

        optimizer = tf.train.GradientDescentOptimizer(0.01)
        train = optimizer.minimize(loss)

    tf.summary.histogram('loss', loss)
    tf.summary.histogram('W', W)
    tf.summary.histogram('b', b)

    init = tf.global_variables_initializer()
    sess = tf.Session()
    sess.run(init)
    merged = tf.summary.merge_all()
    train_writer = tf.summary.FileWriter('./train', sess.graph)

    for i in range(5):
        summary, _ = sess.run([merged, train], feed_dict={input_: features, target: labels})
        train_writer.add_summary(summary, i)



        # curr_W, curr_b, curr_loss = sess.run([W, b, loss], {input_: features, target: labels})
        # print("W: %s b: %s loss: %s" % (curr_W, curr_b, curr_loss))
        # curr_W, curr_b, curr_loss = sess.run([W, b, loss], {input_: features, target: labels})
        # print("W: %s b: %s loss: %s" % (curr_W, curr_b, curr_loss))


def learn2(df):
    # Model parameters
    W = tf.Variable([.3], tf.float32)
    b = tf.Variable([-.3], tf.float32)
    # Model input and output
    x = tf.placeholder(tf.float32)
    linear_model = W * x + b
    y = tf.placeholder(tf.float32)
    # loss
    loss = tf.reduce_sum(tf.square(linear_model - y))  # sum of the squares
    # optimizer
    optimizer = tf.train.GradientDescentOptimizer(0.01)
    train = optimizer.minimize(loss)
    # training data
    x_train = [[1.,2.],[2.,2.],[3.,2.],[4.,2.]]
    y_train = [[1],[2],[3],[4]]

    # Our data example
    # x_train = [[3., 22., 7.25, 1., 0.],
    #            [1., 38., 71.2833, 0., 1.],
    #            [3., 26., 7.925, 0., 1.],
    #            [1., 35., 53.1, 0., 1.],
    #            [3., 35., 8.05, 1., 0.]]
    # y_train = [[0],
    #            [1],
    #            [1],
    #            [1],
    #            [0]]
    # training loop
    init = tf.global_variables_initializer()
    sess = tf.Session()
    sess.run(init)  # reset values to wrong
    for i in range(100):
        sess.run(train, {x: x_train, y: y_train})

    # evaluate training accuracy
    curr_W, curr_b, curr_loss = sess.run([W, b, loss], {x: x_train, y: y_train})
    print("W: %s b: %s loss: %s" % (curr_W, curr_b, curr_loss))


def main():
    df = clean_data("train.csv")
    learn2(df)
    # print(df[['Survived']].as_matrix().transpose())
    # i = 0
    # for row in df.iterrows():
    #     print(row)
    #     i += 1
    #     if i == 4:
    #         break


if __name__ == "__main__":
    main()
