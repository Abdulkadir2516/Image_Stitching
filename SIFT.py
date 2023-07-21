import cv2

# Görüntüyü gri tona dönüştür
image = cv2.imread('data/ornek.png')
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# SIFT nesnesi oluştur ve görüntüdeki özellikleri bul
sift = cv2.SIFT_create()
keypoints = sift.detect(gray, None)

# Görüntüdeki özellikleri çiz
img = cv2.drawKeypoints(gray, keypoints, image)

# Görüntüyü göster
cv2.imshow("SIFT", img)
cv2.waitKey(0)
cv2.destroyAllWindows()