import cv2
import numpy as np

from config import *


def ReadImgAndCvtToBinary(path):
    img = cv2.imread(path, 1)

    blur = cv2.GaussianBlur(img, (5, 5), 0)

    gray = cv2.cvtColor(blur, cv2.COLOR_BGR2GRAY)

    _, binary = cv2.threshold(gray, 127, 255, cv2.THRESH_OTSU | cv2.THRESH_BINARY_INV)

    return np.asarray(binary, dtype=np.float32).reshape([PICTURE_SIZE * PICTURE_SIZE])


def ReadImgandCvtBinaryNoBlur(path):
    img = cv2.imread(path, 1)

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    _, binary = cv2.threshold(gray, 127, 255, cv2.THRESH_OTSU | cv2.THRESH_BINARY_INV)

    return binary
