from dataclasses import dataclass
from math import pi, cos


@dataclass
class ApriltagQuadThreshParams:
    # reject quads containing too few pixels
    min_cluster_pixels: int = 10

    # how many corner candidates to consider when segmenting a group
    # of pixels into a quad.
    max_nmaxima: int = 10

    # Reject quads where pairs of edges have angles that are close to
    # straight or close to 180 degrees. Zero means that no quads are
    # rejected. (In radians).
    # critical_rad: float
    cos_critical_rad: float = cos(10 * pi / 180)

    # When fitting lines to the contours, what is the maximum mean
    # squared error allowed?  This is useful in rejecting contours
    # that are far from being quad shaped; rejecting these quads "early"
    # saves expensive decoding processing.
    max_line_fit_mse: float = 10.0

    # When we build our model of black & white pixels, we add an
    # extra check that the white model must be (overall) brighter
    # than the black model.  How much brighter? (in pixel values,
    # [0,255]). .
    min_white_black_diff: int = 5

    # should the thresholded image be deglitched? Only useful for
    # very noisy images
    deglitch: bool = False
