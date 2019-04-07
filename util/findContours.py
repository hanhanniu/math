import os
import cv2
import numpy as np
from util.predict import PredictModel

min_val = 8
min_range = 30


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
    print("shape before", image.shape)
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
        img = cv2.resize(image, (20, tmp_h), interpolation=cv2.INTER_LINEAR)
        c_w = 4
        c_h = int((28 - tmp_h) / 2)
        re = cv2.copyMakeBorder(img, c_h, c_h, c_w, c_w, cv2.BORDER_CONSTANT, value=0)
    print("shape", re.shape)
    return re


def cutImg(path):
    images = cv2.imread(path)
    blur = cv2.GaussianBlur(images, (5, 5), 3)
    gray = cv2.cvtColor(blur, cv2.COLOR_BGR2GRAY)
    ret, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_OTSU | cv2.THRESH_BINARY_INV)
    cv2.imshow("binary", binary)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    counts = findCounts(binary)
    res = []
    for v in counts:
        tmpimg = binary[v["low"]:v["high"], v["right"]:v["left"]]
        tmpformatimg = formatImg(tmpimg)
        res.append(tmpformatimg)
    return counts, res


# p = PredictModel()
#
# _, res = cutImg(r"D:\PythonPro\mathAI\code\testImgs\easy +\13.jpg")
#
# dat = np.asarray(res).reshape([-1, 784])
#
# print(p.predict(dat))
#
# for v in res:
#     cv2.imshow("u",v)
#     cv2.waitKey(0)
#     cv2.destroyAllWindows()

# filelist = [x for x in os.listdir("test_img")]
#
# for i in filelist:
#     path = os.path.join("test_img", i)
#     img = cv2.imread(path)
#     gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
#     ret, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_OTSU | cv2.THRESH_BINARY)
#     cv2.imshow("i", binary)
#     cv2.waitKey(0)
#     cv2.destroyAllWindows()
#     res = p.predict(np.asarray(binary).reshape([-1, 784]))
#     print("name:%s predict:%s" % (i, res))
