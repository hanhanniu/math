# -*- coding: utf-8 -*-
from config import *
import os
import numpy as np
import cv2

from util.readimg import ReadImgAndCvtToBinary


def ReadImage(type):
    filelist = [x for x in os.listdir(DATASET_URL)]
    num_of_symbol = len(filelist)
    mark = []
    train_total = 0
    evel_total = 0
    for i in range(num_of_symbol):
        characterPath = os.path.join(DATASET_URL, filelist[i])
        filelist2 = [x for x in os.listdir(characterPath) if x.split(".")[-1] == "jpg"]
        characterlength = len(filelist2)
        if type == TYPE_TRAIN:
            m = {
                "num_of_data": int(TRAIN_RATE * characterlength),
                "start_mark": 0
            }
            train_total += int(TRAIN_RATE * characterlength)
            mark.append(m)

        elif type == TYPE_EVEL:
            m = {
                "num_of_data": int(EVEL_RATE * characterlength),
                "start_mark": int(TRAIN_RATE * characterlength)
            }
            evel_total += int(EVEL_RATE * characterlength)
            mark.append(m)
        else:
            raise TypeError("Invalid type")

    images_name = []
    if type == "train":
        images = np.ndarray((train_total, PICTURE_SIZE * PICTURE_SIZE), dtype=np.float32)
        images_lable = np.ndarray((train_total, 17), dtype=np.int32)
    else:
        images = np.ndarray((evel_total, PICTURE_SIZE * PICTURE_SIZE), dtype=np.float32)
        images_lable = np.ndarray((evel_total, 17), dtype=np.int32)

    n = 0
    for i in range(num_of_symbol):
        catpath = os.path.join(DATASET_URL, filelist[i])
        filelist2 = [x for x in os.listdir(catpath) if x.split('.')[-1] == 'jpg']
        for j in range(mark[i]["num_of_data"]):
            repath = os.path.join(catpath, filelist2[j + mark[i]["start_mark"]])
            img = ReadImgAndCvtToBinary(repath)
            images[n] = img
            tpary = np.zeros(17)
            tpary[i] = 1
            images_lable[n] = tpary
            images_name.append(repath)
            n = n + 1
    data = data_batch(images, images_lable, images_name)
    return data


class data_batch(object):

    def __init__(self, images, images_lable, images_name):
        self.images = images
        self.images_lable = images_lable
        self.images_name = images_name
        self.cur_num = 0
        self.length = len(images)

    def next_batch(self, n):
        if self.cur_num >= self.length:
            return
        cnu = self.cur_num + n
        if cnu > self.length:
            cnu = self.length

        img = self.images[self.cur_num:cnu]
        imglab = self.images_lable[self.cur_num: cnu]
        imgna = self.images_name[self.cur_num: cnu]
        self.cur_num = cnu
        return img, imglab, imgna


def Readimg(type):
    filelist = [x for x in os.listdir(DATASET_URL)]
    num_of_symbol = len(filelist)
    mark = []
    train_total = 0
    evel_total = 0
    for i in range(num_of_symbol):
        characterPath = os.path.join(DATASET_URL, filelist[i])
        filelist2 = [x for x in os.listdir(characterPath) if x.split(".")[-1] == "jpg"]
        characterlength = len(filelist2)
        if type == TYPE_TRAIN:
            m = {
                "num_of_data": int(TRAIN_RATE * characterlength),
                "start_mark": 0
            }
            train_total += int(TRAIN_RATE * characterlength)
            mark.append(m)

        elif type == TYPE_EVEL:
            m = {
                "num_of_data": int(EVEL_RATE * characterlength),
                "start_mark": int(TRAIN_RATE * characterlength)
            }
            evel_total += int(EVEL_RATE * characterlength)
            mark.append(m)
        else:
            raise TypeError("Invalid type")

    images_name = []
    if type == "train":
        images = np.ndarray((train_total, PICTURE_SIZE * PICTURE_SIZE), dtype=np.float32)
        images_lable = np.ndarray(train_total, dtype=np.int32)
    else:
        images = np.ndarray((evel_total, PICTURE_SIZE * PICTURE_SIZE), dtype=np.float32)
        images_lable = np.ndarray(evel_total, dtype=np.int32)

    n = 0
    for i in range(num_of_symbol):
        catpath = os.path.join(DATASET_URL, filelist[i])
        filelist2 = [x for x in os.listdir(catpath) if x.split('.')[-1] == 'jpg']
        for j in range(mark[i]["num_of_data"]):
            repath = os.path.join(catpath, filelist2[j + mark[i]["start_mark"]])
            img = ReadImgAndCvtToBinary(repath)
            images[n] = img
            images_lable[n] = j
            images_name.append(repath)
            n = n + 1
    return images,images_lable
