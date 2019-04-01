import tensorflow as tf
import numpy as np
import cv2

from config import *


class CNN(object):

    def __init__(self):
        self.input_x = tf.placeholder(tf.float32, [None, 28 * 28], name="input_x")
        self.input_y = tf.placeholder(tf.int32, [None, 17], name="input_y")
        self.global_step = tf.Variable(0, trainable=False, name="global_step")
        self.keep_prod = tf.placeholder(tf.float32, name="keep_prod")
        self.cnn()

    def cnn(self):
        with tf.name_scope("conv1-relu-maxpool"):
            x_img = tf.reshape(self.input_x, [-1, 28, 28, 1], name="input_img")
            W_conv1 = tf.Variable(tf.truncated_normal([5, 5, 1, 32], stddev=0.1), name="W_conv1")
            b_conv1 = tf.constant(0.1, shape=[32], name="b_conv1")
            conv1 = tf.nn.conv2d(x_img, W_conv1, strides=[1, 1, 1, 1], padding="SAME", name="conv1")
            h1 = tf.nn.relu(tf.nn.bias_add(conv1, b_conv1), name="relu1")

            self.h_pool1 = tf.nn.max_pool(h1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding="SAME",
                                          name="pool1")  # [-1,14,14,32]

        with tf.name_scope("conv2-relu-maxpool"):
            W_conv2 = tf.Variable(tf.truncated_normal([5, 5, 32, 64], stddev=0.1), name="W_conv2")
            b_conv2 = tf.constant(0.1, shape=[64], name="b_conv2")
            conv2 = tf.nn.conv2d(self.h_pool1, W_conv2, strides=[1, 1, 1, 1], padding="SAME", name="conv2")
            h2 = tf.nn.relu(tf.nn.bias_add(conv2, b_conv2), name="relu2")

            self.h_pool2 = tf.nn.max_pool(h2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding="SAME",
                                          name="pool2")  # [-1,7,7,64]

        with tf.name_scope("fc"):
            W_fc1 = tf.Variable(tf.truncated_normal([7 * 7 * 64, 1024], stddev=0.1), name="W_fc1")
            b_fc1 = tf.constant(0.1, shape=[1024], name="b_fc1")
            h_pool2_falt = tf.reshape(self.h_pool2, [-1, 7 * 7 * 64], name="h_pool2_falt")
            h_fc1 = tf.nn.relu(tf.nn.bias_add(tf.matmul(h_pool2_falt, W_fc1), b_fc1), name="h_fc1")

            h_fc1_prod = tf.nn.dropout(h_fc1, self.keep_prod)

            W_fc2 = tf.Variable(tf.truncated_normal([1024, 17], stddev=0.1), name="W_fc2")
            b_fc2 = tf.Variable(tf.constant(0.1, shape=[17]), name="b_fc2")

            #self.logits = tf.nn.relu(tf.matmul(h_fc1_prod, W_fc2) + b_fc2, name="logits")
            self.logits = tf.matmul(h_fc1_prod, W_fc2) + b_fc2
            self.y_pred_cls = tf.argmax(tf.nn.softmax(self.logits), 1)

        with tf.name_scope("loss"):
            cross_entropy = tf.nn.softmax_cross_entropy_with_logits_v2(labels=self.input_y, logits=self.logits,
                                                                       name="cross_entropy")
            self.loss = tf.reduce_mean(cross_entropy)

        with tf.name_scope("optimizer"):
            # optimizer=tf.train.AdamOptimizer(1e-4)
            # grads_and_vars = optimizer.compute_gradients(self.loss)
            # self.optim = optimizer.apply_gradients(grads_and_vars, global_step=self.global_step)
            optimizer = tf.train.AdamOptimizer(1e-4)
            gradients, variables = zip(*optimizer.compute_gradients(self.loss))
            gradients, _ = tf.clip_by_global_norm(gradients, 3.)
            self.optim = optimizer.apply_gradients(zip(gradients, variables), global_step=self.global_step,
                                                   name="optim")

        with tf.name_scope("accuracy"):
            correct_pred = tf.equal(tf.argmax(self.input_y, 1), self.y_pred_cls, name="correct_pred")
            self.acc = tf.reduce_mean(tf.cast(correct_pred, tf.float32))
