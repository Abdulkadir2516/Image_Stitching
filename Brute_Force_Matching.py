import numpy as np
import cv2
from matplotlib import pyplot as plt

img1 = cv2.imread('./data/image_left.jpg',0)
img2 = cv2.imread('./data/image_right.jpg',0)

sift = cv2.SIFT_create()

kp1, des1 = sift.detectAndCompute(img1,None)
kp2, des2 = sift.detectAndCompute(img2,None)

bf = cv2.BFMatcher()
matches = bf.knnMatch(des1,des2, k=2)

good = []
for m,n in matches:
    if m.distance < 0.06*n.distance:
        good.append([m])

img3 = cv2.drawMatchesKnn(img1,kp1,img2,kp2,good,outImg=None,flags=2)

plt.imshow(img3),plt.show()
