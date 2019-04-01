import os
import time
import tensorflow as tf

from config import *
from cnn_model import CNN
from util.load_data import InputData


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
        for i in range(10000):
            trainimg, lab = data.train.next_batch(count_size)
            if i % 100 == 0:
                mr, loss, step = sess.run([merged_summary, model.loss, model.global_step],
                                          feed_dict={model.input_x: trainimg, model.input_y: lab, model.keep_prod: 1})
                writer.add_summary(mr, global_step=step)
                train_acc = model.acc.eval(feed_dict={model.input_x: trainimg, model.input_y: lab, model.keep_prod: 1})
                print('step {},loss {},accuracy: {}'.format(i, loss, train_acc))
                eloss, eacc = evel(sess, data)
                print("step {},evel_loss {},evel_acc {}\n".format(i, eloss, eacc))
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
    print("evel acc of cnn>>>>")
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
