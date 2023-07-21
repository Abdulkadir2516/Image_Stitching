import cv2
import numpy as np

# Görüntüyü yükle
img = cv2.imread('data/ornek.jpg', 0)

# Gauss filtrelerini ayarla
sigma1 = 1
sigma2 = 5
ksize1 = int(4 * sigma1 + 1)
ksize2 = int(4 * sigma2 + 1)

# Görüntüye filtreleri uygula
blur1 = cv2.GaussianBlur(img, (ksize1, ksize1), sigma1)
blur2 = cv2.GaussianBlur(img, (ksize2, ksize2), sigma2)

# DoG algoritmasını uygula
dog = cv2.absdiff(blur1, blur2)

# Görüntüyü göster
cv2.imshow('DoG Algoritması', dog)
cv2.waitKey(0)
cv2.destroyAllWindows()
