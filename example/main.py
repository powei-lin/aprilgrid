import numpy as np
import cv2
from aprilgrid import Detector
from glob import glob

if __name__ == '__main__':
    file_list = sorted(glob("dataset-calib-cam1_1024_16/mav0/cam0/data/*.png"))
    # file_list = sorted(glob("dataset-calib-cam1_512_16/mav0/cam0/data/*.png"))
    # file_list = sorted(glob("example/*.jpg"))
    detector = Detector('t36h11')
    for i, file_name in enumerate(file_list):
        print(i)
        # if i < 76:
        #     continue
        img = cv2.imread(file_name, cv2.IMREAD_GRAYSCALE)
        # img = cv2.resize(img, (952, 1264), None)
        img_color = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        detections = detector.detect(img)
        for detection in detections:
            center = np.round(np.average(detection.corners, axis=0)).astype(np.int32)
            cv2.putText(img_color, f"{detection.tag_id}", center[0], 1, 1, (0, 0, 255))
            for j, c in enumerate(detection.corners):
                c = np.round(c[0]).astype(np.int32)
                cv2.circle(img_color, c, 3, (0, 255, 0))


        # print(img.shape)
        # print(im_max.shape)
        # img_color = cv2.resize(img_color, None, None, 0.3, 0.3)
        cv2.imshow("im", img_color)
        # cv2.imshow("im_max", im_max)
        cv2.waitKey(1)