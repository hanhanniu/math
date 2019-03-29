import os
import time
from tensorflow.examples.tutorials.mnist import input_data
import tensorflow as tf
import numpy as np
import cv2

from config import *
from cnn_model import CNN
from util.imgInput import ReadImage
from util.load_data import InputData


# def train_evel():
#     model = CNN()
#     data=InputData()
#
#     count_size = 50
#     count_times = int(data.train.length/ count_size) + 1
#     print(count_times)
#
#     start_time = time.time()
#     tf.summary.scalar("loss", model.loss)
#     tf.summary.scalar("accuracy", model.acc)
#     merged_summary = tf.summary.merge_all()
#     writer = tf.summary.FileWriter(TENSORBOARD_DIR)
#     saver = tf.train.Saver()
#     init=tf.global_variables_initializer()
#     with tf.Session() as sess:
#         sess.run(init)
#         writer.add_graph(sess.graph)
#         for i in range(count_times):
#             train, lab= data.train.next_batch(count_size)
#
#             _, global_step, train_summaryes, train_loss, train_accuracy = sess.run(
#                 [model.optim, model.global_step, merged_summary, model.loss, model.acc],
#                 feed_dict={model.input_x: train, model.input_y: lab, model.keep_prod: 0.4})
#
#             if global_step % 100 == 0:
#                 s = sess.run(merged_summary, feed_dict={model.input_x: train, model.input_y: lab, model.keep_prod: 1})
#                 writer.add_summary(s, global_step=global_step)
#
#                 print("global_step:%d\t loss:%f\t acc:%f" % (global_step, train_loss, train_accuracy))
#
#         saver.save(sess, save_path=SAVE_DIR)
#     end_time = time.time()
#     print("train takes %d Seconds" % (int(end_time) - int(start_time)))


# def train_evel():
#     model = CNN()
#     # data=input_data.read_data_sets("./mnist",one_hot=True)
#     data = InputData()
#     count_size = 50
#
#     start_time = time.time()
#     tf.summary.scalar("loss", model.loss)
#     tf.summary.scalar("accuracy", model.acc)
#     merged_summary = tf.summary.merge_all()
#     writer = tf.summary.FileWriter(TENSORBOARD_DIR)
#     saver = tf.train.Saver()
#     init = tf.global_variables_initializer()
#     with tf.Session() as sess:
#         sess.run(init)
#         writer.add_graph(sess.graph)
#         for i in range(3000):
#             train, lab = data.train.next_batch(count_size)
#
#             _, global_step, train_summaryes, train_loss, train_accuracy = sess.run(
#                 [model.optim, model.global_step, merged_summary, model.loss, model.acc],
#                 feed_dict={model.input_x: train, model.input_y: lab, model.keep_prod: 0.4})
#
#             if global_step % 100 == 0:
#                 s = sess.run(merged_summary, feed_dict={model.input_x: train, model.input_y: lab, model.keep_prod: 1})
#                 writer.add_summary(s, global_step=global_step)
#
#                 print("global_step:%d\t loss:%f\t acc:%f" % (global_step, train_loss, train_accuracy))
#
#         saver.save(sess, save_path=SAVE_DIR)
#     end_time = time.time()
#     print("train takes %d Seconds" % (int(end_time) - int(start_time)))


def train():
    # model = CNN()
    # data=input_data.read_data_sets("./mnist",one_hot=True)
    data = InputData()
    count_size = 50

    start_time = time.time()

    # 配置Tensorboard时将存在的tensorboard文件删除，否则会覆盖
    for i in os.listdir(TENSORBOARD_DIR):
        p = os.path.join(TENSORBOARD_DIR, i)
        os.remove(p)

    tf.summary.scalar("loss", model.loss)
    tf.summary.scalar("accuracy", model.acc)
    merged_summary = tf.summary.merge_all()
    writer = tf.summary.FileWriter(TENSORBOARD_DIR)
    saver = tf.train.Saver()
    init = tf.global_variables_initializer()
    with tf.Session() as sess:
        sess.run(init)
        writer.add_graph(sess.graph)
        best_acc = 0
        best_step = 0
        for i in range(4500):
            trainimg, lab = data.train.next_batch(count_size)
            if i % 100 == 0:
                mr, loss, step = sess.run([merged_summary, model.loss, model.global_step],
                                          feed_dict={model.input_x: trainimg, model.input_y: lab, model.keep_prod: 1})
                writer.add_summary(mr, global_step=step)
                train_acc = model.acc.eval(feed_dict={model.input_x: trainimg, model.input_y: lab, model.keep_prod: 1})
                print('setp {},loss {},accuracy: {}'.format(i, loss, train_acc))
                eloss, eacc = evel(sess, data)
                if eacc > best_acc:
                    best_acc = eacc
                    best_step = i
                    saver.save(sess, save_path=SAVE_DIR)
            model.optim.run(feed_dict={model.input_x: trainimg, model.input_y: lab, model.keep_prod: 0.7})
        print("best_step:{}".format(best_step))
        a = []
        for i in range(350):
            timg, tlab = data.test.next_batch(count_size)
            test_acc = model.acc.eval(feed_dict={model.input_x: timg, model.input_y: tlab, model.keep_prod: 1})
            a.append(test_acc)
        print("test acc:{}".format(sum(a) / 350))
        # saver.save(sess, save_path=SAVE_DIR)
    end_time = time.time()
    print("train takes %d Seconds" % (int(end_time) - int(start_time)))


def evel(sess, data):
    a = []
    b = []
    for i in range(350):
        timg, tlab = data.test.next_batch(50)
        loss, acc = sess.run([model.loss, model.acc],
                             feed_dict={model.input_x: timg, model.input_y: tlab, model.keep_prod: 1})
        a.append(loss)
        b.append(acc)
    return sum(a) / 350, sum(b) / 350


if __name__ == "__main__":
    model = CNN()
    train()
