import numpy as np
# from common import image_u8_decimate, max_pool
import cv2
# from apriltag import Detector

arr = np.random.randint(1, 10, (7, 4)).astype(np.uint8)
print(arr)
d = cv2.dilate(arr, np.ones((3, 3)))
print(d)
# max_pool(arr, 2)
# d = Detector()
# img = cv2.imread("001.jpg", cv2.IMREAD_GRAYSCALE)
# im_max = d.threshold(img)
# print(img.shape)
# print(im_max.shape)
# cv2.imshow("im", img)
# cv2.imshow("im_max", im_max)
# cv2.waitKey(0)
# print(arr)
# m, n = arr.shape
# print(arr.reshape(m//2, 2, n//2, 2))
# pooled = arr.reshape(m//2, 2, n//2, 2).max((1, 3))
# print(pooled)

# block_size = (2, 3)
# num_blocks = (7, 5)
# arr_shape = np.array(block_size) * np.array(num_blocks)
# numel = arr_shape.prod()
# arr = np.random.randint(1, numel, numel).reshape(arr_shape)
# print(arr.shape)

# m, n = arr.shape  # pretend we only have this
# pooled = arr.reshape(m//block_size[0], block_size[0],
#                     n//block_size[1], block_size[1]).max((1, 3))

# img = cv2.imread("april_grid_800mm.png", cv2.IMREAD_GRAYSCALE)
# img = cv2.resize(img, None, None, 0.3, 0.3)
# print(img.shape)
# img_small = image_u8_decimate(img, 1.5)
# print(img_small.shape)
# cv2.imshow("img", img_small)
# cv2.waitKey(0)
