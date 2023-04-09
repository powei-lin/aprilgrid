import numpy as np
import cv2
from aprilgrid import Detector
from aprilgrid.common import random_color
from glob import glob
from PIL import Image
from pillow_avif import AvifImagePlugin
import matplotlib.colors as mcolors

if __name__ == '__main__':
    file_list = sorted(glob("cam0/data/*.png"))
    # file_list = sorted(glob("dataset-calib-cam1_512_16/mav0/cam0/data/*.png"))
    # file_list = sorted(glob("example/data/*.jpg"))
    detector = Detector('t36h11')
    count = 0
    imgs = []
    print()
    colors = [np.array(mcolors.to_rgb(c))*255 for c in mcolors.TABLEAU_COLORS.values()]
    colors = [c.astype(np.uint8).tolist() for c in colors]
    print(colors)
    # exit()
    color_dict = {}
    for i, file_name in enumerate(file_list):
        img = cv2.imread(file_name, cv2.IMREAD_GRAYSCALE)
        img = cv2.resize(img, (1200, 900), None)
        img_color = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        detections = detector.detect(img)
        ss = f"frame {i}, detect {len(detections)} tags"
        cv2.putText(img_color, ss, (20, 30), 2, 1, (255, 0, 0))
        count += len(detections)
        for detection in detections:
            center = np.round(np.average(detection.corners, axis=0)).astype(np.int32)
            cv2.putText(img_color, f"{detection.tag_id}", center[0], 5, 1, (255, 200, 0), 2)
            for j, c in enumerate(detection.corners):
                c = np.round(c[0]).astype(np.int32)
                id = j
                # cv2.putText(img_color, f"{id}", c, 5, 1, (0, 0, 255))
                cv2.circle(img_color, c, 3, colors[id], -1)
        imgs.append(Image.fromarray(img_color))


        # print(img.shape)
        # print(im_max.shape)
        # img_color = cv2.resize(img_color, None, None, 0.5, 0.5)
        cv2.imshow("im", img_color)
        # cv2.imshow("im_max", im_max)
        cv2.waitKey(1)
        if i > 240:
            break
    imgs[0].save(
                "example.avif",
                save_all=True,
                append_images=imgs[1:],
                duration=80
            )
    print(f"avg: {count/len(file_list):.3f} tags")