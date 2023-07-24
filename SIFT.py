import cv2

image = cv2.imread('data/ornek.png')
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# SIFT nesnesi ile görüntüdeki anahtar noktalarının tespiti
sift = cv2.SIFT_create()
keypoints = sift.detect(gray, None)

# Görüntüdeki anahtar noklaranın çizimi
img = cv2.drawKeypoints(gray, keypoints, image)

cv2.imshow("SIFT", img)
cv2.waitKey(0)
cv2.destroyAllWindows()