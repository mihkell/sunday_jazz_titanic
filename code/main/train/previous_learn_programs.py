

def learn_matrix_multiplication_model(df):
    # Model parameters
    W = tf.Variable(tf.ones([5, 1]), tf.float32)
    b = tf.Variable(tf.ones(1), tf.float32)
    # Model input and output
    x = tf.placeholder(tf.float32, shape=(None, 5))
    y = tf.placeholder(tf.float32, shape=(None, 1))

    linear_model = tf.matmul(x, W) + b
    loss = tf.reduce_sum(tf.abs(linear_model - y))
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


def sigmoid_accuracy(df):
    # Model parameters
    W = tf.Variable(tf.ones([5, 1]), tf.float32)
    b = tf.Variable(tf.ones(1), tf.float32)
    # Model input and output
    x = tf.placeholder(tf.float32, shape=(None, 5))
    y = tf.placeholder(tf.float32, shape=(None, 1))

    linear_model = tf.nn.sigmoid(tf.matmul(x, W) + b)
    loss = tf.reduce_sum(tf.abs(linear_model - y))

    accuracy = tf.abs(linear_model - y)  # reduce_count_zeros(linear_model - y < threshold) / len(y)
    accuracy = tf.count_nonzero(accuracy < 0.2) / 714
    # optimizer
    optimizer = tf.train.GradientDescentOptimizer(0.01)
    train = optimizer.minimize(loss)

    # Our data example
    x_train, y_train = get_features_labels(df)
    # training loop
    # accuracy = tf.contrib.metrics.accuracy(y, linear_model)
    init = tf.global_variables_initializer()
    with tf.Session() as sess:
        sess.run(init)
        range_val = 1000
        batch_size = 10
        for i in range(range_val):
            for batch_start in range(0, len(x_train) - batch_size, batch_size):
                batch_end = batch_start + batch_size
                features = x_train[batch_start:batch_end]
                labels = y_train[batch_start:batch_end]
                sess.run([train], {x: features, y: labels})

        curr_W, curr_b, curr_loss, prediction = sess.run([W, b, loss, accuracy], {x: x_train, y: y_train})
        print("Accuracy %s " % prediction)
        print("W: %s b: %s loss: %s" % (curr_W, curr_b, curr_loss))
        # evaluate training accuracy
