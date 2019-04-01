import math

import cv2
import numpy as np
import matplotlib.pyplot as plt

from util.readimg import ReadImgAndCvtToBinary

min_val = 10
min_range = 30


# def findOneConture(img, type):
#     ary = np.asarray(img)
#     h, w = ary.shape[:2]
#     #print("height:%d,width:%d" % (h, w))
#     s = []
#     # 计算像素密度
#     if type == "height":
#         for i in range(h):
#             avg = sum(ary[i]) / w
#             s.append(avg)
#     else:
#         for i in range(w):
#             avg = sum(ary[:, i]) / h
#             s.append(avg)
#     print(len(s))
#     # 根据计算的像素密度画出直方图
#     # if type=="height":
#     #     x = [x for x in range(h)]
#     #     y = [s[y] for y in range(h)]
#     # else:
#     #     x = [x for x in range(w)]
#     #     y = [s[y] for y in range(w)]
#     #
#     # plt.plot(x, y)
#     # plt.show()
#     w = []
#     dmax = np.asarray(s).argmax()
#     # print("max:%d" % dmax)
#     i = 0
#     while i < len(s) - 3:
#         if s[i] > 0.2 * dmax:
#             w.append(i)
#         i = i + 3
#     # print(w)
#     re = []
#     re.append(w[0])
#     for i in range(len(w) - 1):
#         if w[i + 1] - w[i] == 3:
#             continue
#         else:
#             re.append(w[i])
#             re.append(w[i + 1])
#     re.append(w[-1])
#     # print(re)
#     result = []
#     j = 0
#     while j < len(re) - 1:
#         result.append((re[j], re[j + 1]))
#         j = j + 2
#     return result
#
#
# def findContours(img):
#     rows = findOneConture(img, 'height')
#     #print(rows)
#     result = []
#     for i in range(len(rows)):
#         cols = findOneConture(img[rows[i][0]:rows[i][1]], 'width')
#         for j in range(len(cols)):
#             result.append({"low": rows[i][0], "high": rows[i][1], "left": cols[j][0], "right": cols[j][1]})
#     return result

def findOneCount(data):
    start_i = None
    end_i = None
    peek_ranges = []
    for i, val in enumerate(data):
        if val > min_val and start_i is None:
            start_i = i
        elif val > min_val and start_i is not None:
            pass
        elif val < min_val and start_i is None:
            pass
        elif val < min_val and start_i is not None:
            if i - start_i > min_range:
                end_i = i
                peek_ranges.append((start_i, end_i))
                start_i = None
                end_i = None
        else:
            raise ValueError("can't pass this case")
    return peek_ranges


def findCounts(image):
    h, w = image.shape[:2]
    print("width", w)
    print("height", h)
    img_data = np.asarray(image, dtype=np.float32)
    high_sum = np.sum(img_data, axis=1) / w
    peek_ranges = findOneCount(high_sum)
    result = []
    for val in peek_ranges:
        line_img = img_data[val[0]:val[1], :]
        width_sum = np.sum(line_img, axis=0) / h
        peeks = findOneCount(width_sum)
        for v in peeks:
            result.append({"low": val[0], "high": val[1], "right": v[0], "left": v[1]})
    return result


def formatImg(image):
    h, w = image.shape[:2]
    if h > w:
        i = h / 20
        tmp_w = int(w / i)
        if tmp_w % 2 == 1:
            tmp_w = tmp_w + 1
        img = cv2.resize(image, (tmp_w, 20), interpolation=cv2.INTER_LINEAR)
        c_h = 4
        c_w = int((28 - tmp_w) / 2)
        re = cv2.copyMakeBorder(img, c_h, c_h, c_w, c_w, cv2.BORDER_CONSTANT, value=0)
    else:
        i = w / 20
        tmp_h = int(h / i)
        if tmp_h % 2 == 1:
            tmp_h = tmp_h + 1
        img = cv2.resize(image, (tmp_h, 20), interpolation=cv2.INTER_LINEAR)
        c_w = 4
        c_h = int((28 - tmp_h) / 2)
        re = cv2.copyMakeBorder(img, c_h, c_h, c_w, c_w, cv2.BORDER_CONSTANT, value=0)
    return re


img = cv2.imread(r"C:\Users\admin\Desktop\hell\9.jpg")

gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

ret, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)
#
# cv2.imshow("img",binary)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

result = findCounts(binary)

i1 = result[0]
i2 = result[1]
i3 = result[2]

tmp_img=binary[i1["low"]:i1["high"], i1["right"]:i1["left"]]

rre=formatImg(tmp_img)

cv2.imshow("img1", rre)
cv2.waitKey(0)
cv2.destroyAllWindows()

print(len(result))
print(result)
