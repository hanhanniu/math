import cv2

import tensorflow as tf
import os
import numpy as np
from cnn_model import CNN, SAVE_DIR
from util.findContours import findContours
from util.predict import PredictModel

# a = tf.Variable(3)
# saver = tf.train.Saver(tf.global_variables(), )
# model=CNN()
# with tf.Session()as sess:
#     sess.run(tf.global_variables_initializer())
#     saver.save(sess, os.path.abspath(SAVE_DIR))
p = PredictModel()

img = cv2.imread(r"C:\Users\admin\Desktop\hell\5.png", 1)

gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

blur = cv2.GaussianBlur(gray, (13, 13), 0)

ret, binary = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)

_, binary2 = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)
# binary=cv2.adaptiveThreshold(gray,255,cv2.ADAPTIVE_THRESH_MEAN_C,cv2.THRESH_BINARY,5,0)
# print(binary)

cv2.imshow("binary", binary)
cv2.waitKey(0)
cv2.destroyAllWindows()

ibin=np.asarray(binary).reshape([-1,784])
res=p.predict(ibin)
print(res)

count = findContours(binary)
# print(count)
# print(len(count))

w, h = binary.shape[:2]

a = []
for i in count:
    low = i["low"]
    high = i["high"]
    left = i["left"]
    right = i["right"]
    if low - 10 >= 0:
        low = low - 10
    if high + 10 <= h:
        high = high + 10
    if left - 10 >= 0:
        left = left - 10
    if right + 10 <= w:
        right = right + 10
    # tmp = img[i["low"]:i["high"], i["left"]:i["right"]]
    tmp = binary2[low:high, left:right]
    tmp2 = cv2.resize(tmp, (28, 28), interpolation=cv2.INTER_LINEAR)
    # print(tmp2.shape)
    # g = cv2.cvtColor(tmp2, cv2.COLOR_BGR2GRAY)
    # ret, b = cv2.threshold(g, 127, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)
    # print(b)
    cv2.imshow("binary", tmp2)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    tn = np.asarray(tmp2).reshape([784])
    a.append(tn)

re = p.predict(a)

print(re)
