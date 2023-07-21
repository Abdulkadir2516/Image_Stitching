import cv2
import numpy as np

img_ = cv2.imread('./data/image_left.jpg')
img_ = cv2.resize(img_, (0,0), fx=1, fy=1)
img1 = cv2.cvtColor(img_,cv2.COLOR_BGR2GRAY)

img = cv2.imread('./data/image_right.jpg')
img = cv2.resize(img, (0,0), fx=1, fy=1)
img2 = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

sift = cv2.SIFT_create()
# find the key points and descriptors with SIFT
kp1, des1 = sift.detectAndCompute(img1,None)
kp2, des2 = sift.detectAndCompute(img2,None)

cv2.imshow('image_left_keypoints Sift',cv2.drawKeypoints(img_,kp1,None))

# Surf and ORB Example
# orb = cv2.ORB_create()
# # find the key points and descriptors with SIFT
# kp1, des1 = orb.detectAndCompute(img1,None)
# kp2, des2 = orb.detectAndCompute(img2,None)
#
# cv2.imshow('image_left_keypoints ORB',cv2.drawKeypoints(img_,kp1,None))

# surf = cv2.SURF_create()
# # find the key points and descriptors with SIFT
# kp1, des1 = surf.detectAndCompute(img1,None)
# kp2, des2 = surf.detectAndCompute(img2,None)

# cv2.imshow('image_left_keypoints Surf',cv2.drawKeypoints(img_,kp1,None))



cv2.waitKey(0)
cv2.destroyAllWindows()