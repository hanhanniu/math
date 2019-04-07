import os

from cnn_model import CNN
import tensorflow as tf
from config import *


class PredictModel(object):

    def __init__(self):
        self.session = tf.Session()
        self.model = CNN()
        self.session.run(tf.global_variables_initializer())

        saver = tf.train.Saver()
        saver.restore(sess=self.session, save_path=SAVE_DIR_MNIST)

    def predict(self, img_data):
        pred = self.session.run(self.model.y_pred_cls,
                                feed_dict={self.model.input_x: img_data, self.model.keep_prod: 1})
        print("predict num", pred)
        result = ""
        for i in pred:
            result += SYMBOL[i]
        return result
