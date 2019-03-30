import csv
import numpy as np


class Data(object):

    def __init__(self, filepath):
        self.images = []
        self.labels = []
        self.offset = 0
        self.length = 0
        self.read_data(filepath)

    def read_data(self, filepath):
        with open(filepath, "r", encoding="utf-8", newline="") as input_file:
            reader = csv.reader(input_file)
            for line in reader:
                temp = np.asarray(line, dtype=np.int32)
                self.images.append(temp[1:])
                i = temp[0]
                n = np.zeros([17])
                n[i] = 1
                self.labels.append(n)
            self.length = len(self.images)

    def next_batch(self, count):
        n = self.offset + count
        if n > self.length:
            n = self.offset + count - self.length
            imgs = self.images[self.offset:]
            labs = self.labels[self.offset:]
            tmpimgs = self.images[0:n]
            tmplabs = self.labels[0:n]
            if len(imgs) == 0:
                imgs = tmpimgs
                labs = tmplabs
            elif len(tmpimgs) != 0 and len(tmplabs) != 0:
                imgs = np.r_[imgs, tmpimgs]
                labs = np.r_[labs, tmplabs]
            self.offset = n
        else:
            imgs = self.images[self.offset:n]
            labs = self.labels[self.offset:n]
            self.offset = n
        return imgs, labs


class InputData(object):

    def __init__(self):
        self.train = Data("../dataset/train.csv")
        self.test = Data("../dataset/test.csv")
