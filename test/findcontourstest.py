import cv2
import numpy as np

from util.findContours import findContours

img = cv2.imread(r"C:\Users\admin\Desktop\hell\textcount.png", 1)

gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

blur = cv2.GaussianBlur(gray, (13, 13), 0)

ret, binary = cv2.threshold(blur, 127, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)
# binary=cv2.adaptiveThreshold(gray,255,cv2.ADAPTIVE_THRESH_MEAN_C,cv2.THRESH_BINARY,5,0)

# cv2.imshow("binary", binary)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

count = findContours(binary)
print(count)
print(len(count))

# t = count[14]

# cv2.imshow("jie", img[t["low"]-5:t["high"]+5, t["left"]-5:t["right"]+5])
# cv2.imshow("jie",img[42:129,102:141])
# cv2.waitKey(0)
# cv2.destroyAllWindows()

# for i in range(len(count)):
#     t = img[count[i]["low"]:count[i]["high"], count[i]["left"]:count[i]["right"]]
#     name = "../static/test_{}.jpg".format(i)
#     cv2.imwrite(name, t)
