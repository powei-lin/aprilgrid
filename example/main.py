import numpy as np
import cv2
from aprilgrid import Detector
from glob import glob

if __name__ == "__main__":
    file_list = sorted(glob("example/data/*.jpg"))
    detector = Detector("t36h11")
    count = 0
    for i, file_name in enumerate(file_list):
        img = cv2.imread(file_name, cv2.IMREAD_GRAYSCALE)
        img_color = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        detections = detector.detect(img)
        print(f"frame {i}, detect {len(detections)} tags")
        count += len(detections)
        for detection in detections:
            center = np.round(np.average(detection.corners, axis=0)).astype(np.int32)
            cv2.putText(img_color, f"{detection.tag_id}", center[0], 2, 2, (0, 0, 255))
            for j, c in enumerate(detection.corners):
                c = np.round(c[0]).astype(np.int32)
                id = detection.tag_id * 4 + j
                cv2.putText(img_color, f"{id}", c, 2, 2, (0, 0, 255))
                cv2.circle(img_color, c, 3, (0, 255, 0))

        img_color = cv2.resize(img_color, None, None, 0.5, 0.5)
        cv2.imshow("im", img_color)
        cv2.waitKey(1)
    print(f"avg: {count / len(file_list):.3f} tags")
