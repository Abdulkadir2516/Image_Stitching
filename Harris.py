import cv2

# Görüntüyü yükle
img = cv2.imread('data/ornek.jpg')

# Görüntüyü gri tona dönüştür
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Harris köşe detektörü parametreleri
block_size = 2  # Kare yama boyutu
ksize = 3  # Sobel çekirdeği boyutu
k = 0.04  # Harris skorunun duyarlılık parametresi

# Harris köşe detektörü uygula
harris_corners = cv2.cornerHarris(gray, block_size, ksize, k)

# Köşeleri işaretle
img[harris_corners > 0.01 * harris_corners.max()] = [0, 0, 255]

# Görüntüyü göster
cv2.imshow('Harris Kesif Algoritması', img)
cv2.waitKey(0)
cv2.destroyAllWindows()