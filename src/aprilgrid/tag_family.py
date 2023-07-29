from dataclasses import dataclass
import numpy as np
from typing import List
from .tag_codes import APRILTAG_CODE_DICT
from .detection import Detection
import cv2


@dataclass
class TagFamily:
    marker_edge: int
    border_bit: int
    min_distance: int
    _hamming_thres: int = 3
    debug_level: int = 0

    def __post_init__(self):
        self.name = f"t{self.marker_edge**2}h{self.min_distance}"
        if self.name not in APRILTAG_CODE_DICT:
            raise ValueError(
                f"{self.name} is not in {APRILTAG_CODE_DICT.keys()}")
        self.tag_bit_list = np.array([np.array([bool(int(i)) for i in np.binary_repr(
            tag, 36)]) for tag in APRILTAG_CODE_DICT[self.name]])

        self.marker_edge_bit = 2 * self.border_bit + self.marker_edge  # tagFamily.d 10
        edge_position = self.marker_edge_bit - 0.5
        self.tag_corners = np.expand_dims(np.array(
            [[-0.5, -0.5], [edge_position, -0.5], [edge_position, edge_position], [-0.5, edge_position]], np.float32), 1)

    def decode(self, detect_code: np.ndarray, quad, detections: List[Detection]):
        code_mat = detect_code.copy()
        for r in range(4):
            scores = np.count_nonzero(
                code_mat.flatten() != self.tag_bit_list, axis=1)
            best_score_idx = np.argmin(scores)
            best_score = scores[best_score_idx]
            if best_score < self._hamming_thres:
                # print(f"best tag: {best_score_idx}, hamming: {best_score}")
                new_quad = np.flip(np.roll(quad, -r, axis=0), axis=0)
                detections.append(Detection(best_score_idx, new_quad))
                if self.debug_level > 0:
                    print(f"detect {best_score_idx} rotate {r} time")
                return
            else:
                code_mat = np.rot90(code_mat)

    def decodeQuad(self, quads, gray: np.ndarray) -> List[Detection]:
        """
        decode the Quad
        :param quads: array of quad which have four points
        :param gray: gray picture
        :return: array of detection
        """
        detections = []
        # points = []
        # whitepoint = []
        # h, w = gray.shape

        for quad in quads:
            H, _ = cv2.findHomography(quad, self.tag_corners)

            tag_img = cv2.warpPerspective(
                gray, H, (self.marker_edge_bit, self.marker_edge_bit))
            if self.debug_level > 0:
                cv2.imshow("debug single tag", tag_img)
                cv2.waitKey(0)

            avg_brightness = np.average(tag_img)
            # TODO add some filter
            detect_code = np.where(tag_img[self.border_bit:-self.border_bit,
                                   self.border_bit: -self.border_bit] > avg_brightness+20, True, False)
            self.decode(detect_code, quad, detections)
        return detections


TAG_FAMILY_DICT = {
    "t36h11": TagFamily(6, 2, 11, 3),
    "t36h11b1": TagFamily(6, 1, 11, 3),
    "t25h9": TagFamily(5, 2, 9, 2),
    "t25h9b1": TagFamily(5, 1, 9, 2),
    "t25h7": TagFamily(5, 2, 7, 2),
    "t25h7b1": TagFamily(5, 1, 7, 2),
    "t16h5": TagFamily(4, 2, 5, 1),
    "t16h5b1": TagFamily(4, 1, 5, 1),
}
