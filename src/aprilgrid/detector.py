from dataclasses import dataclass
import numpy as np
import cv2
from typing import List
from .tag_family import TAG_FAMILY_DICT
from .qtp import ApriltagQuadThreshParams
from .detection import Detection
from .common import max_pool, random_color


@dataclass
class Detector:
    tag_family_name: str
    quad_decimate = 2.0
    quad_sigma = 0.0
    refine_edges: bool = True
    decode_sharpening: float = 0.25

    def __post_init__(self):
        self.tag_family = TAG_FAMILY_DICT[self.tag_family_name]
        self.qtp = ApriltagQuadThreshParams()
    
    def detect(self, img: np.ndarray) -> List[Detection]:
        # step 1 resize
        max_size = np.max(img.shape)
        im_blur = cv2.GaussianBlur(img, (3, 3), 1)
        im_blur_resize = im_blur.copy()
        new_size_ratio = 1
        if max_size > 1000:
            new_size_ratio = 1000.0 / max_size
            im_blur_resize = cv2.resize(im_blur_resize, None, None, new_size_ratio, new_size_ratio)
        
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
        h, w = im.shape[0], im.shape[1]

        im_copy = im.copy()

        threshim = self.threshold(im_copy)
        # cv2.imshow("threshim", threshim)
        # threshim = cv2.GaussianBlur(threshim, (3, 3), 1)
        (cnts, _) = cv2.findContours(threshim,
                                     cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)

        # debug
        # output = np.zeros((h, w, 3), dtype=np.uint8)
        # for c in cnts:
        #     cv2.drawContours(output, [c], -1, random_color(), 2)
        #     cv2.imshow("debug", output)
        #     cv2.waitKey(0)

        cnts = [c for c in cnts if (c.shape[0] >= 4)]
        quads = []  # array of quad including four peak points
        for c in cnts:
            area = cv2.contourArea(c)
            if area > self.qtp.min_cluster_pixels:
                hull = cv2.convexHull(c)
                areahull = cv2.contourArea(hull)
                # debug
                # cv2.drawContours(output, [c], -1, random_color(), 2)
                # cv2.imshow("debug", output)
                # cv2.waitKey(0)
                if (area / areahull > 0.8):
                    # maximum_area_inscribed
                    quad = cv2.approxPolyDP(hull, 8, True)
                    if (len(quad) == 4):
                        areaqued = cv2.contourArea(quad)
                        if areaqued / areahull > 0.8 and areahull >= areaqued:
                            # Calculate the refined corner locations
                            quads.append(quad)
        # cv2.drawContours(output, quads, -1, (0, 255, 0), 2)
        # cv2.imshow("debug", output)
        # cv2.waitKey(0)
        return quads

    def threshold(self, im: np.ndarray) -> np.ndarray:
        h, w = im.shape

        tilesz = 4
        im_max = max_pool(im, tilesz, True)
        im_min = max_pool(im, tilesz, False)

        # small_im = cv2.resize(im, None, None, 0.25, 0.25)
        kernel0 = np.ones((3, 3), dtype=np.uint8)
        im_max = cv2.dilate(im_max, kernel0)
        # # im_max = cv2.resize(im_max, (w, h))
        # kernel1 = np.ones((3, 3), dtype=np.uint8)
        im_min = cv2.erode(im_min, kernel0)
        # im_min = cv2.resize(im_min, (w, h))
        im_min = np.repeat(np.repeat(im_min, tilesz, axis=1), tilesz, axis=0)
        im_max = np.repeat(np.repeat(im_max, tilesz, axis=1), tilesz, axis=0)
        
        edge = max(h%tilesz, w%tilesz)
        im_min = np.pad(im_min, (0, edge), 'edge')[:h, :w]
        im_max = np.pad(im_max, (0, edge), 'edge')[:h, :w]

        im_diff = im_max-im_min
        # im_diff = cv2.resize(im_diff, (w, h))
        # dd = np.where(diff < self.qtp.min_white_black_diff, 127, im_min)
        threshim = np.where(im_diff < self.qtp.min_white_black_diff, np.uint8(0),
                            np.where(im > (im_min + im_diff // 2), np.uint8(255), np.uint8(0)))
        
        # hires can try dilate twice
        threshim = cv2.dilate(threshim, kernel0)
        return threshim

        
        
    def threshold_old(self, im: np.ndarray) -> np.ndarray:
        w = im.shape[1]
        h = im.shape[0]

        assert (w < 32768)
        assert (h < 32768)

        threshim = np.zeros((h, w), dtype=np.uint8)
        # assert(threshim->stride == s);

        # The idea is to find the maximum and minimum values in a
        # window around each pixel. If it's a contrast-free region
        # (max-min is small), don't try to binarize. Otherwise,
        # threshold according to (max+min)/2.
        #
        # Mark low-contrast regions with value 127 so that we can skip
        # future work on these areas too.

        # however, computing max/min around every pixel is needlessly
        # expensive. We compute max/min for tiles. To avoid artifacts
        # that arise when high-contrast features appear near a tile
        # edge (and thus moving from one tile to another results in a
        # large change in max/min value), the max/min values used for
        # any pixel are computed from all 3x3 surrounding tiles. Thus,
        # the max/min sampling area for nearby pixels overlap by at least
        # one tile.
        #
        # The important thing is that the windows be large enough to
        # capture edge transitions; the tag does not need to fit into
        # a tile.

        # XXX Tunable. Generally, small tile sizes--- so long as they're
        # large enough to span a single tag edge--- seem to be a winner.
        tilesz = 4

        # the last (possibly partial) tiles along each row and column will
        # just use the min/max value from the last full tile.
        tw = w // tilesz
        th = h // tilesz

        #  first, collect min/max statistics for each tile
        im_max = max_pool(im, tilesz, True)
        im_min = max_pool(im, tilesz, False)

        # second, apply 3x3 max/min convolution to "blur" these values
        # over larger areas. This reduces artifacts due to abrupt changes
        # in the threshold value.
        kernel = np.ones((3, 3), dtype=np.uint8)
        if True:
            im_max = cv2.dilate(im_max, kernel)
            im_min = cv2.erode(im_min, kernel)
        im_min = np.repeat(np.repeat(im_min, tilesz, axis=1), tilesz, axis=0)
        im_max = np.repeat(np.repeat(im_max, tilesz, axis=1), tilesz, axis=0)

        im_diff = im_max - im_min
        threshim = np.where(im_diff < self.qtp.min_white_black_diff, np.uint8(127),
                            np.where(im > (im_min + im_diff // 2), np.uint8(255), np.uint8(0)))
        threshim = cv2.dilate(threshim, kernel)
        # threshim = np.where( im_diff < self.qtp.min_white_black_diff, np.uint8(0), im)
        # debug
        # print(threshim.dtype)
        # print(im_diff.shape)
        # print(im.shape)
        # cv2.imshow("tt", threshim)
        # cv2.waitKey(0)

        # # we skipped over the non-full-sized tiles above. Fix those now.
        # if True:
        #     for (int y = 0; y < h; y++) {

        #         # what is the first x coordinate we need to process in this row?

        #         int x0;

        #         if (y >= th*tilesz) {
        #             x0 = 0; // we're at the bottom; do the whole row.
        #         } else {
        #             x0 = tw*tilesz; // we only need to do the right most part.
        #         }

        #         // compute tile coordinates and clamp.
        #         int ty = y / tilesz;
        #         if (ty >= th)
        #             ty = th - 1;

        #         for (int x = x0; x < w; x++) {
        #             int tx = x / tilesz;
        #             if (tx >= tw)
        #                 tx = tw - 1;

        #             int max = im_max[ty*tw + tx];
        #             int min = im_min[ty*tw + tx];
        #             int thresh = min + (max - min) / 2;

        #             uint8_t v = im->buf[y*s+x];
        #             if (v > thresh)
        #                 threshim->buf[y*s+x] = 255;
        #             else
        #                 threshim->buf[y*s+x] = 0;
        #         }
        #     }
        # }

        return threshim
