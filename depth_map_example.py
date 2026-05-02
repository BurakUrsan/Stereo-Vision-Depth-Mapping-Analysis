import cv2
import matplotlib.pyplot as plt

# Sol ve sağ görüntüleri gri tonlamalı yükle
imgL = cv2.imread('sol_goruntu.jpg', 0)
imgR = cv2.imread('sag_goruntu.jpg', 0)

stereo = cv2.StereoBM_create(numDisparities=64, blockSize=15)

# Derinlik (disparity) haritasını hesapla
disparity = stereo.compute(imgL, imgR)

# Sonucu görselleştir
plt.imshow(disparity, 'gray')
plt.title('Derinlik Haritası (Disparity Map)')
plt.show()
