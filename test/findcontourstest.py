import os

import cv2
import numpy as np
import tensorflow as tf

from cnn_model import CNN
from util.findContours import findContours
from util.predict import PredictModel

img = cv2.imread(r"C:\Users\admin\Desktop\hell\9.jpg", 1)

gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

blur = cv2.GaussianBlur(gray, (13, 13), 0)

ret, binary = cv2.threshold(blur, 127, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)
# binary=cv2.adaptiveThreshold(gray,255,cv2.ADAPTIVE_THRESH_MEAN_C,cv2.THRESH_BINARY,5,0)
# print(binary)

# cv2.imshow("binary", binary)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

count = findContours(binary)
# print(count)
# print(len(count))

a = []
for i in count:
    tmp = img[i["low"]:i["high"], i["left"]:i["right"]]
    tmp2 = cv2.resize(tmp, (28, 28), interpolation=cv2.INTER_LINEAR)
    # print(tmp2.shape)
    g = cv2.cvtColor(tmp2, cv2.COLOR_BGR2GRAY)
    ret, b = cv2.threshold(g, 127, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    # cv2.imshow("binary", b)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    tn = np.asarray(b).reshape([784])
    a.append(tn)
#
# model = CNN()
# with tf.Session()as sess:
#     saver = tf.train.Saver()
#     sess.run(tf.global_variables_initializer())
#     saver.restore(sess, os.path.abspath("../util/save_dir"))
#     sess.run(model.y_pred_cls, feed_dict={model.input_x: a, model.keep_prod: 1})

p = PredictModel()

re = p.predict(a)
print(re)

# t = count[14]

# cv2.imshow("jie", img[t["low"]-5:t["high"]+5, t["left"]-5:t["right"]+5])
# cv2.imshow("jie",img[42:129,102:141])
# cv2.waitKey(0)
# cv2.destroyAllWindows()

# for i in range(len(count)):
#     t = img[count[i]["low"]:count[i]["high"], count[i]["left"]:count[i]["right"]]
#     name = "../static/test_{}.jpg".format(i)
#     cv2.imwrite(name, t)
