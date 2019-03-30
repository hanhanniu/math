import math

import cv2
import numpy as np
import matplotlib.pyplot as plt


def findOneConture(img, type):
    ary = np.asarray(img)
    h, w = ary.shape[:2]
    #print("height:%d,width:%d" % (h, w))
    s = []
    # 计算像素密度
    if type == "height":
        for i in range(h):
            avg = sum(ary[i]) / w
            s.append(avg)
    else:
        for i in range(w):
            avg = sum(ary[:, i]) / h
            s.append(avg)
    print(len(s))
    # 根据计算的像素密度画出直方图
    # if type=="height":
    #     x = [x for x in range(h)]
    #     y = [s[y] for y in range(h)]
    # else:
    #     x = [x for x in range(w)]
    #     y = [s[y] for y in range(w)]
    #
    # plt.plot(x, y)
    # plt.show()
    w = []
    dmax = np.asarray(s).argmax()
    # print("max:%d" % dmax)
    i = 0
    while i < len(s) - 3:
        if s[i] > 0.2 * dmax:
            w.append(i)
        i = i + 3
    # print(w)
    re = []
    re.append(w[0])
    for i in range(len(w) - 1):
        if w[i + 1] - w[i] == 3:
            continue
        else:
            re.append(w[i])
            re.append(w[i + 1])
    re.append(w[-1])
    # print(re)
    result = []
    j = 0
    while j < len(re) - 1:
        result.append((re[j], re[j + 1]))
        j = j + 2
    return result


def findContours(img):
    rows = findOneConture(img, 'height')
    #print(rows)
    result = []
    for i in range(len(rows)):
        cols = findOneConture(img[rows[i][0]:rows[i][1]], 'width')
        for j in range(len(cols)):
            result.append({"low": rows[i][0], "high": rows[i][1], "left": cols[j][0], "right": cols[j][1]})
    return result
