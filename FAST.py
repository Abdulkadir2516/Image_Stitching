import numpy as np
import cv2

img = cv2.imread('data/ornek.jpg',0)

# Initiate FAST object with default values
fast = cv2.FastFeatureDetector_create()

# find and draw the keypoints
kp = fast.detect(img,None)
img2 = cv2.drawKeypoints(img, kp, outImage=None, color=(255,0,0))

# Görüntüyü göster
cv2.imshow("FAST", img2)
cv2.waitKey(0)
cv2.destroyAllWindows()