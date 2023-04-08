from dataclasses import dataclass
import numpy as np
import cv2

@dataclass
class Tag_family:
    _d: int
    _blackBorder: int

    def decodeQuad(self, quads, gray: np.ndarray):
        """
        decode the Quad
        :param quads: array of quad which have four points
        :param gray: gray picture
        :return: array of detection
        """
        detections = []
        points = []
        whitepoint = []
        h, w = gray.shape
        marker_edge_bit = 2 * self._blackBorder + self._d  # tagFamily.d 10
        edge_position = marker_edge_bit-0.5
        corners2 = np.expand_dims(np.array([[edge_position, -0.5], [edge_position, edge_position], [-0.5, edge_position], [-0.5, -0.5]], np.float32), 1)
        for quad in quads[14:]:
            debug_img = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
            print(quad.shape)
            H, _ = cv2.findHomography(quad, corners2)
            debug_img = cv2.warpPerspective(gray, H, (marker_edge_bit, marker_edge_bit))
            avg = np.average(debug_img)
            detect_code = np.where(debug_img[self._blackBorder:-self._blackBorder, self._blackBorder: -self._blackBorder] > avg, True, False)
            # self._decode(detect_code, quad)
            print(detect_code)
            print(avg)
            # print(debug_img.shape)
            # print(H)
            # cv2.drawContours(debug_img, [quad.astype(np.int32)],  -1, np.random.randint(0, 255, 3, np.uint8).tolist(), 1)
            cv2.imshow("debug", debug_img)
            cv2.waitKey(0)