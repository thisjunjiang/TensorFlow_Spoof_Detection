## this version uses labels_onehot and softmax

import pandas as pd
import numpy as np
import tensorflow as tf
import time

values = np.load('dil3_values.npy')
labels = np.load('dil3_labels.npy')
labels = labels.reshape(247500,1)

labels_onehot = tf.one_hot(labels,2)
labels_onehot = labels_onehot.eval(session = tf.Session())
labels_onehot = labels_onehot.reshape(247500,2)

def add_layer(inputs, in_size, out_size, n_layer, activation_function=None):
    # add one more layer and return the output of this layer
    layer_name = 'layer%s' % n_layer
    with tf.name_scope(layer_name):
        with tf.name_scope('weights'):
            Weights = tf.Variable(tf.random_normal([in_size, out_size]), name='W')
            tf.summary.histogram(layer_name + '/weights', Weights)
        with tf.name_scope('biases'):
            biases = tf.Variable(tf.zeros([1, out_size]) + 0.1, name='b')
            tf.summary.histogram(layer_name + '/biases', biases)
        with tf.name_scope('Wx_plus_b'):
            Wx_plus_b = tf.add(tf.matmul(inputs, Weights), biases)
        if activation_function is None:
            outputs = Wx_plus_b
        else:
            outputs = activation_function(Wx_plus_b, )
        tf.summary.histogram(layer_name + '/outputs', outputs)
        return Weights, biases, outputs

## para
hidden_layers = 1
hidden_units = 10
n_input = 4
n_classes = 2
learning_rate = 0.8

## define network
with tf.name_scope('inputs'):
    xs = tf.placeholder(tf.float32, [None, n_input], name="x_input")
    ys = tf.placeholder(tf.float32, [None, n_classes], name="y_intput")

# add hidden layer
W1, b1, l1 = add_layer(xs, n_input, hidden_units, 1, activation_function=tf.nn.tanh)
# add output layer
W2, b2, prediction = add_layer(l1, hidden_units, n_classes, 2, activation_function=tf.nn.softmax)

with tf.name_scope('loss'):
    ## coss and train step
    # cross_entropy = tf.reduce_mean(-tf.reduce_sum(ys * tf.log(prediction)))
    cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=prediction, labels=ys))
    tf.summary.scalar('cross_entropy', cross_entropy)

with tf.name_scope('train'):
    train_step = tf.train.GradientDescentOptimizer(learning_rate).minimize(cross_entropy)

with tf.name_scope('accuracy'):
    ## accuracy
    correct_prediction = tf.equal(tf.argmax(prediction, 1), tf.argmax(ys, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    tf.summary.scalar('accuracy', accuracy)

with tf.Session() as sess:
    st = time.time()
    merged = tf.summary.merge_all()
    writer = tf.summary.FileWriter("logs/3", graph=sess.graph)
    init = tf.global_variables_initializer()
    sess.run(init)
    for i in range(501):
        sess.run(train_step, feed_dict = {xs:values, ys:labels_onehot})
        if i%50 == 0:
            print('accuracy: ', sess.run(accuracy, feed_dict = {xs:values, ys:labels_onehot}))
            print('cross_entropy: ', sess.run(cross_entropy, feed_dict = {xs:values, ys:labels_onehot}))
            result = sess.run(merged, feed_dict = {xs:values, ys:labels_onehot})
            writer.add_summary(result, i)
    writer.close()
    end = time.time()
    print('*' * 30)
    print('training finish. cost time:', int(end-st) , 'seconds; accuracy:', sess.run(accuracy, feed_dict={xs: values, ys: labels_onehot}))

