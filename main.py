import numpy as np
from common import image_u8_decimate
import cv2

img = cv2.imread("april_grid_800mm.png", cv2.IMREAD_GRAYSCALE)
img = cv2.resize(img, None, None, 0.3, 0.3)
print(img.shape)
img_small = image_u8_decimate(img, 1.5)
print(img_small.shape)
cv2.imshow("img", img_small)
cv2.waitKey(0)
