from dataclasses import dataclass
import numpy as np

@dataclass
class Detector:
    nthreads = 1
    quad_decimate = 2.0
    quad_sigma = 0.0

    def detect(self, img: np.ndarray):       
        pass
