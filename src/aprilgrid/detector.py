from dataclasses import dataclass
import numpy as np
import cv2
from typing import List
from .tag_family import TAG_FAMILY_DICT
from .detection import Detection
from .common import max_pool, random_color


@dataclass
class Detector:
    tag_family_name: str
    quad_decimate = 2.0
    quad_sigma = 0.0
    refine_edges: bool = True
    decode_sharpening: float = 0.25
    min_white_black_diff: int = 5
    debug_level: int = 0

    def __post_init__(self):
        self.tag_family = TAG_FAMILY_DICT[self.tag_family_name]
        self.min_cluster_pixels = self.tag_family.marker_edge_bit**2

    def detect(self, img: np.ndarray) -> List[Detection]:
        # step 1 resize
        max_size = np.max(img.shape)
        im_blur = cv2.GaussianBlur(img, (3, 3), 1)
        im_blur_resize = im_blur.copy()
        new_size_ratio = 1
        if max_size > 1000:
            new_size_ratio = 1000.0 / max_size
            im_blur_resize = cv2.resize(
                im_blur_resize, None, None, new_size_ratio, new_size_ratio)

        # detect quads
        quads = self.apriltag_quad_thresh(im_blur_resize)

        # refine corner
        winSize = (5, 5)
        zeroZone = (-1, -1)
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TermCriteria_COUNT, 40, 0.001)

        # refine on small image
        if new_size_ratio < 1:
            quads = [cv2.cornerSubPix(im_blur_resize, quad.astype(
                np.float32), winSize, zeroZone, criteria) for quad in quads]
            quads = [quad/new_size_ratio for quad in quads]

        # refine on oringinal image
        quads = [cv2.cornerSubPix(img, quad.astype(
            np.float32), winSize, zeroZone, criteria) for quad in quads]
        detections = self.tag_family.decodeQuad(quads, img)
        return detections

    def apriltag_quad_thresh(self, im: np.ndarray):
        # step 1. threshold the image, creating the edge image.

        im_copy = im.copy()

        threshim = self.threshold(im_copy)
        (cnts, _) = cv2.findContours(threshim,
                                     cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)

        # debug
        if self.debug_level > 0:
            h, w = im.shape[0], im.shape[1]
            output = np.zeros((h, w, 3), dtype=np.uint8)
            if self.debug_level == 1:
                for c in cnts:
                    cv2.drawContours(output, [c], -1, random_color(), 2)
                    cv2.imshow("debug", output)
                    cv2.waitKey(0)

        cnts = [c for c in cnts if (c.shape[0] >= 4)]
        quads = []  # array of quad including four peak points
        for c in cnts:
            area = cv2.contourArea(c)
            if area > self.min_cluster_pixels:
                hull = cv2.convexHull(c)
                areahull = cv2.contourArea(hull)
                # debug
                if self.debug_level == 2:
                    cv2.drawContours(output, [c], -1, random_color(), 2)
                    cv2.imshow("debug", output)
                    cv2.waitKey(0)
                if (area / areahull > 0.8):
                    # maximum_area_inscribed
                    quad = cv2.approxPolyDP(hull, 8, True)
                    if (len(quad) == 4):
                        areaqued = cv2.contourArea(quad)
                        if areaqued / areahull > 0.8 and areahull >= areaqued:
                            # Calculate the refined corner locations
                            quads.append(quad)
        return quads

    def threshold(self, im: np.ndarray) -> np.ndarray:
        h, w = im.shape

        tilesz = 4
        im_max = max_pool(im, tilesz, True)
        im_min = max_pool(im, tilesz, False)

        kernel0 = np.ones((3, 3), dtype=np.uint8)
        im_max = cv2.dilate(im_max, kernel0)
        im_min = cv2.erode(im_min, kernel0)
        im_min = np.repeat(np.repeat(im_min, tilesz, axis=1), tilesz, axis=0)
        im_max = np.repeat(np.repeat(im_max, tilesz, axis=1), tilesz, axis=0)

        edge = max(h % tilesz, w % tilesz)
        im_min = np.pad(im_min, (0, edge), 'edge')[:h, :w]
        im_max = np.pad(im_max, (0, edge), 'edge')[:h, :w]

        im_diff = im_max-im_min
        threshim = np.where(im_diff < self.min_white_black_diff, np.uint8(0),
                            np.where(im > (im_min + im_diff // 2), np.uint8(255), np.uint8(0)))

        # hi-res img can try dilate twice
        threshim = cv2.dilate(threshim, kernel0)
        return threshim
