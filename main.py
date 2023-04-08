import numpy as np
# from common import image_u8_decimate, max_pool
from common import timeit
import cv2
from apriltag import Detector
# a = np.array([True])
# b = np.array([[True], [False]])
# print(a != b)
# exit()
# [format(t, "0{}b".format(code_size)) for t in tag_codes]

# x = np.array([[1, 3], [2, 4]])
# print(x)
# print(np.repeat(np.repeat(x, 2, axis=1), 2, axis=0))
# exit()


# arr = np.random.randint(1, 10, (7, 4)).astype(np.uint8)
# print(arr)
# print(np.where(arr > 3, arr, 0))
# exit()
# d = cv2.dilate(arr, np.ones((3, 3)))
# print(d)
# max_pool(arr, 2)
d = Detector()
img = cv2.imread("001.jpg", cv2.IMREAD_GRAYSCALE)
img = cv2.resize(img, (952, 1264), None)
img_color = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
detections = timeit(d.detect)(img)
exit()
for (id, quad) in detections:
    center = np.round(np.average(quad, axis=0)).astype(np.int32)

    cv2.putText(img_color, f"{id}", center[0], 1, 1, (0, 0, 255))


# print(img.shape)
# print(im_max.shape)
# cv2.imshow("im", img_color)
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
