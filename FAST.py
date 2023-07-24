import numpy as np
import cv2

img1 = cv2.imread('./data/image_left.jpg',0) # queryImage


fast = cv2.FastFeatureDetector_create()

# görüntüdeki anahtar noktalarının tespiti ve çizimi
kp = fast.detect(img1,None)
img2 = cv2.drawKeypoints(img1, kp, outImage=None, color=(255,0,0))

cv2.imshow("FAST", img2)
cv2.waitKey(0)
cv2.destroyAllWindows()