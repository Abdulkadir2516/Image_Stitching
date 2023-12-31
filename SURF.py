import cv2

img = cv2.imread('data/ornek.jpg')

img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

surf = cv2.SURF_create()
# OpenCV'nin 4.0.0 sürümünden itibaren, SURF algoritması OpenCV'nin ana dağıtımından çıkarılmıştır.

keypoints, descriptors = surf.detectAndCompute(img, None)

img = cv2.drawKeypoints(img, keypoints, None)

cv2.imshow('SURF', img)

cv2.waitKey(0)
cv2.destroyAllWindows()