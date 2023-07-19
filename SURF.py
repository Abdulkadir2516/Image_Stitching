import cv2


img = cv2.imread('ornek.jpg')


img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

surf = cv2.xfeatures2d.SURF_create()

keypoints, descriptors = surf.detectAndCompute(img, None)

img = cv2.drawKeypoints(img, keypoints, None)

cv2.imshow('SURF', img)

cv2.waitKey(0)
cv2.destroyAllWindows()