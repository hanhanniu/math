import config
from util.imgInput import ReadImage
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import cv2

data = ReadImage(config.TYPE_EVEL)

print(len(data.images))

count_size = 3
count_times = int(len(data.images) / count_size) + 1

count = 0
for i in range(count_times):
    img, lab, name = data.next_batch(count_size)
    for j in range(len(img)):
        count = count + 1
        print("count:%d\t name:%s\t lab:%s\t sum:%d" % (count, name[j], lab[j], np.asarray(img[j]).sum()))

# b = tf.reshape(img, [-1, 45, 45, 1])
# w = tf.Variable(tf.truncated_normal([5, 5, 1, 1], stddev=0.1))
#
# with tf.Session() as sess:
#     tf.global_variables_initializer().run()
#     conv = tf.nn.conv2d(b, w, strides=[1, 1, 1, 1], padding="SAME")
#     print(sess.run(conv))

# print(len(img[0]))
#
# r = np.asarray(img[10]).reshape((45, 45))
# print(r)
#
# plt.imshow(r, cmap='Greys', interpolation='None')
# plt.title('image')  # 图像题目
# plt.show()
