import os

import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelBinarizer

pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 100000)
resource_path = '../../data/'
file_name = 'picled_data.pickle'

SEX = 'Sex'
AGE = 'Age'
CLASS = 'Pclass'
FARE = 'Fare'
AGE_MISSING = 'AgeMissing'


def normalize(df):
    AGE = 'Age'
    ageMax = df[AGE].max()
    ageMin = df[AGE].min()
    df[AGE] = (df[AGE] - ageMin) / (ageMax - ageMin)
    df[AGE] = (df[AGE] ** 0.8 * 2 - 1)

    FARE = 'Fare'
    fareMax = df[FARE].max()
    fareMin = df[FARE].min()
    df[FARE] = (df[FARE] - fareMin) / (fareMax - fareMin)
    df[FARE] = df[FARE] * 2 - 1

    return df


def clean_data(data_filename):
    file_path = resource_path + file_name
    # if not os.path.exists(file_path):
    df = pd.read_csv('../../data/' + data_filename)

    # df = df[[CLASS, AGE, FARE, SEX]] # FIXME add 'Survived'
    df = df[['Survived', CLASS, AGE, FARE, SEX]] # FIXME add 'Survived'

    df[AGE_MISSING] = df[AGE].isnull()
    df[AGE_MISSING] = LabelBinarizer().fit_transform(df[AGE_MISSING]) * 2 - 1

    df = df.fillna(method='bfill')
    df = df.fillna(method='ffill')

    df[SEX] = LabelBinarizer().fit_transform(df[SEX])
    df[SEX] = df[SEX] * 2 - 1

    df = normalize(df)

    pd.to_pickle(df, file_path)
    return pd.read_pickle(file_path)


def get_features_labels(data):
    return data[[SEX, CLASS, AGE, AGE_MISSING, FARE]].as_matrix(), data[['Survived']].as_matrix()


def learn(df, test_set):
    # Try next time:
    # http://stackoverflow.com/questions/33919948/how-to-set-adaptive-learning-rate-for-gradientdescentoptimizer

    x_train, y_train = get_features_labels(df)
    x_test, y_test = get_features_labels(test_set)

    learning_rate = tf.placeholder(tf.float32)
    inputs = tf.placeholder(tf.float32, shape=(None, x_train.shape[1]), name='inputs')
    labels = tf.placeholder(tf.float32, shape=(None, 1))
    # make forward pass of layer
    layer_sigmoid = tf.contrib.layers.fully_connected(inputs,
                                                      3,
                                                      activation_fn=tf.nn.softplus,
                                                      weights_initializer=tf.random_normal_initializer(seed=1))

    layer_sigmoid = tf.contrib.layers.fully_connected(layer_sigmoid, 20,
                                                      activation_fn=tf.nn.softplus,
                                                      weights_initializer=tf.random_normal_initializer(seed=1))
    layer_sigmoid = tf.contrib.layers.fully_connected(layer_sigmoid, 10,
                                                      activation_fn=tf.nn.softplus,
                                                      weights_initializer=tf.random_normal_initializer(seed=1))

    logits = tf.contrib.layers.fully_connected(layer_sigmoid,
                                               1,
                                               activation_fn=tf.sigmoid,
                                               weights_initializer=tf.random_normal_initializer(seed=1))
    result = tf.placeholder(tf.float32, shape=(None, 1), name='result')
    result = logits * 1.0

    # accuracy
    accuracy = tf.abs(logits - labels)
    accuracy = tf.count_nonzero(accuracy < 0.5) / tf.placeholder(tf.int64, name='input_size')
    tf.summary.scalar('accuracy', accuracy)

    # training
    # survived = [logits[i] for i in range(labels.shape[0]) if labels[i][0] == tf.constant(1., dtype=tf.float32)]

    # loss = tf.reduce_sum(tf.abs(logits - labels))
    loss = (tf.reduce_sum(labels * (1 - logits)) / (tf.reduce_sum(labels) + 0.0001) + \
            tf.reduce_sum((1 - labels) * logits) / (tf.reduce_sum(1 - labels) + 0.0001)) * 0.5
    tf.summary.scalar('loss', 1 - loss)
    optimizer = tf.train.AdadeltaOptimizer(learning_rate)#, global_step=tf.placeholder(tf.int64,
    # name='count'))
    train = optimizer.minimize(loss)

    # running graph
    with tf.Session() as sess:
        saver = tf.train.Saver()
        # For visualization
        merged = tf.summary.merge_all()
        train_writer = tf.summary.FileWriter('summary/train', sess.graph)
        test_writer = tf.summary.FileWriter('summary/test')
        sess.run(tf.global_variables_initializer())
        batch_size = 10
        count = 1
        for i in range(3):  # 2 rounds of training
            for batch_start in range(0, len(x_train) - batch_size, batch_size):
                count += 1
                batch_end = batch_start + batch_size
                features = x_train[batch_start:batch_end]
                labels_ = y_train[batch_start:batch_end]
                _, merged_summary = sess.run([train, merged], {inputs: features,
                                                               labels: labels_,
                                                               learning_rate: 0.01,
                                                               'input_size:0': labels_.shape[0]
                                                               #'count:0': count
                                                               })
                train_writer.add_summary(merged_summary, count)

        ran_loss_val, accuracy_val = sess.run([loss, accuracy],
                                              {inputs: x_test, labels: y_test,
                                               'input_size:0': y_test.shape[0]})
        print(i, '1 - loss/accuracy:', 1 - ran_loss_val, accuracy_val)
        saver.save(sess, 'my_test_model')


def compute_validation():
    validation_data = clean_data('test.csv')

    sess = tf.Session()
    new_saver = tf.train.import_meta_graph('my_test_model.meta')
    new_saver.restore(sess, tf.train.latest_checkpoint('./'))

    sess.run(tf.global_variables_initializer())


    graph = tf.get_default_graph()
    w1 = graph.get_tensor_by_name("input_size:0")
    # w1 = graph.get_tensor_by_name("inputs:0")
    # w2 = graph.get_tensor_by_name("logits:0")

    print(sess.run(w1, {'input_size:0': validation_data.shape[0]}))

def main():
    data = clean_data("train.csv")
    print('data-size-total:', data.shape[0])
    train_set, test_test = train_test_split(data, random_state=1)

    learn(train_set, test_test)


if __name__ == "__main__":
    main()
    # compute_validation()
