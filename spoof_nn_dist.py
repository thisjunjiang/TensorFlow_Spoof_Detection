## this version uses labels_onehot and softmax
# coding=utf-8
import pandas as pd
import numpy as np
import tensorflow as tf
import time

# Define parameters for distribution
FLAGS = tf.app.flags.FLAGS
tf.app.flags.DEFINE_string("job_name", "", "One of 'ps', 'worker'")
tf.app.flags.DEFINE_integer("task_index", 0, "Index of task within the job")

def main(_):
    #cluster = tf.train.ClusterSpec({"ps": ["tf-node01:2222"], "worker": ["tf-node04:2224", "tf-node05:2225", "tf-node06:2226"]})
    cluster = tf.train.ClusterSpec({"ps": ["tf-node01:2222"], "worker": ["tf-node04:2224", "tf-node05:2225"]})
    server = tf.train.Server(cluster,job_name=FLAGS.job_name,task_index=FLAGS.task_index)

    if FLAGS.job_name == 'ps':
        server.join()
    elif FLAGS.job_name == 'worker':
        with tf.device(tf.train.replica_device_setter(worker_device = '/job:worker/task:%d' % FLAGS.task_index, cluster = cluster)):
            global_step = tf.Variable(0, name = 'global_step', trainable = False)

            ## define inputs
            with tf.name_scope('inputs'):
                xs = tf.placeholder(tf.float32, [None, 4], name="x_input")
                ys = tf.placeholder(tf.float32, [None, 2], name="y_intput")


            # add hidden layer
            W1, b1, l1 = add_layer(xs, 4, 10, 1, activation_function=tf.nn.tanh)
            # add output layer
            W2, b2, prediction = add_layer(l1, 10, 2, 2, activation_function=tf.nn.softmax)

            with tf.name_scope('cross_entropy'):
                cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=prediction, labels=ys))
                tf.summary.scalar('cross_entropy', cross_entropy)

            with tf.name_scope('accuracy'):
                ## accuracy
                correct_prediction = tf.equal(tf.argmax(prediction, 1), tf.argmax(ys, 1))
                accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
                tf.summary.scalar('accuracy', accuracy)

            with tf.name_scope('train'):
                optimizer = tf.train.GradientDescentOptimizer(0.8)
                grads_and_vars = optimizer.compute_gradients(cross_entropy)

                train_step = optimizer.apply_gradients(grads_and_vars, global_step = global_step)

            init = tf.global_variables_initializer()

            saver = tf.train.Saver()
            merged = tf.summary.merge_all()

            sv = tf.train.Supervisor(is_chief=(FLAGS.task_index == 0),
                                     logdir="./checkpoint/",
                                     init_op=init,
                                     summary_op=merged,
                                     saver=saver,
                                     global_step=global_step,
                                     save_model_secs=5)
            values = np.load('dil3_values.npy')
            labels = np.load('dil3_labels.npy')
            labels = labels.reshape(247500,1)

            labels_onehot = np.load('dil3_labels_onehot.npy')
            with sv.prepare_or_wait_for_session(server.target) as sess:
                step = 0

                st = time.time()
                while step < 1001:
                    _, ac_, cr_, step = sess.run([train_step, accuracy, cross_entropy, global_step], feed_dict = {xs:values, ys:labels_onehot})

                    if step%50 == 0:
                        ac, cr = sess.run([accuracy, cross_entropy], feed_dict={xs: values, ys: labels_onehot})
                        print('step: %d, accuracy: %f, cross_entropy: %f' %(step, ac, cr))
                end = time.time()
                print('*' * 30)
                print('training finish. cost time:', end-st , 'seconds; accuracy:', sess.run(accuracy, feed_dict={xs: values, ys: labels_onehot}))
            #sv.stop()


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

if __name__ == "__main__":
    tf.app.run()
