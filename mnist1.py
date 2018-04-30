from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# Imports
import numpy as np
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

# the model is: y = softmax(W*x + b)

LOG_DIR = './summaries'


def main():
    testWriter = tf.summary.FileWriter(LOG_DIR + '/test')
    mnist = input_data.read_data_sets("./MNIST-data", one_hot=True)

    x = tf.placeholder(tf.float32, [None, 784])
    W = tf.Variable(tf.zeros([784, 10]))
    b = tf.Variable(tf.zeros([10]))
    y = tf.matmul(x, W) + b
    y_ = tf.placeholder(tf.float32, [None, 10])
    print(y_, y)

    loss = tf.reduce_mean(
        tf.nn.softmax_cross_entropy_with_logits(
            labels=y_,
            logits=y
        )
    )
    train_step = tf.train.GradientDescentOptimizer(0.5).minimize(loss)

    sess = tf.InteractiveSession()
    tf.global_variables_initializer().run()

    # train the model
    for (i) in range(200):
        batch_xs, batch_ys = mnist.train.next_batch(100)
        sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys})

        with tf.name_scope('test'):
            correct_prediction = tf.equal(
                tf.argmax(y, 1),
                tf.argmax(y_, 1)
            )
            test_accuracy = tf.reduce_mean(tf.cast(
                correct_prediction,
                tf.float32
            ))
            foo = tf.summary.scalar('accuracy', test_accuracy)
            merged = tf.summary.merge_all()

        # test the model
        if i % 10 == 0:
            summary, acc = sess.run(
                [merged, test_accuracy],
                feed_dict={x: mnist.test.images,
                           y_: mnist.test.labels}
            )
            print(acc)
            testWriter.add_summary(summary, i)


if __name__ == "__main__":
    main()
