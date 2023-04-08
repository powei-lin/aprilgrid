import numpy as np
import cv2
from aprilgrid import Detector

if __name__ == '__main__':

    detector = Detector('t36h11')
    img = cv2.imread("example/001.jpg", cv2.IMREAD_GRAYSCALE)
    img = cv2.resize(img, (952, 1264), None)
    img_color = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    detections = detector.detect(img)
    for (id, quad) in detections:
        center = np.round(np.average(quad, axis=0)).astype(np.int32)

        cv2.putText(img_color, f"{id}", center[0], 1, 1, (0, 0, 255))


    # print(img.shape)
    # print(im_max.shape)
    cv2.imshow("im", img_color)
    # cv2.imshow("im_max", im_max)
    cv2.waitKey(0)