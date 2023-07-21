import cv2

img = cv2.imread('data/ornek.png', cv2.IMREAD_GRAYSCALE)

orb = cv2.ORB_create()

keypoints, descriptors = orb.detectAndCompute(img, None)

img = cv2.drawKeypoints(img, keypoints, None)

cv2.imshow('ORB', img)

cv2.waitKey(0)
cv2.destroyAllWindows()

