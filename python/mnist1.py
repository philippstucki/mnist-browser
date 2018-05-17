from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import os
import shutil

# the model used is: y = softmax(W*x + b)
LOG_DIR = './summaries'
SAVED_DIR = './saved'


def main():
    mnist = input_data.read_data_sets("./MNIST-data", one_hot=True)
    sess = tf.InteractiveSession()

    x = tf.placeholder(tf.float32, [None, 784], name='x')
    W = tf.Variable(tf.zeros([784, 10]))
    b = tf.Variable(tf.zeros([10]))
    y = tf.add(tf.matmul(x, W), b, name='y')
    y_ = tf.placeholder(tf.float32, [None, 10])

    loss = tf.reduce_mean(
        tf.nn.softmax_cross_entropy_with_logits_v2(
            labels=y_,
            logits=y
        )
    )
    train_step = tf.train.GradientDescentOptimizer(
        learning_rate=0.5
    ).minimize(loss)

    tf.global_variables_initializer().run()
    trainWriter = tf.summary.FileWriter(LOG_DIR + '/train', sess.graph)
    testWriter = tf.summary.FileWriter(LOG_DIR + '/test', sess.graph)

    correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
    accuracy = tf.reduce_mean(tf.cast(
        correct_prediction, tf.float32
    ))

    tf.summary.scalar('accuracy', accuracy)
    tf.summary.histogram('weights', W)
    tf.summary.histogram('biases', b)
    tf.summary.scalar('loss', loss)
    merged = tf.summary.merge_all()

    for (i) in range(201):
        batch_xs, batch_ys = mnist.train.next_batch(100)
        summary_train, acc = sess.run([merged, train_step], feed_dict={
                                      x: batch_xs, y_: batch_ys})
        trainWriter.add_summary(summary_train, i)

        # every nth iteration test the model
        if i % 10 == 0:
            summary, acc = sess.run(
                [merged, accuracy],
                feed_dict={x: mnist.test.images,
                           y_: mnist.test.labels}
            )
            print(acc)
            testWriter.add_summary(summary, i)

    if os.path.exists(SAVED_DIR):
        shutil.rmtree(SAVED_DIR)

    tf.saved_model.simple_save(
        session=sess,
        export_dir=SAVED_DIR,
        inputs={"x": x},
        outputs={"y": y}
    )

    [print(n.name) for n in tf.get_default_graph().as_graph_def().node]


if __name__ == "__main__":
    main()
