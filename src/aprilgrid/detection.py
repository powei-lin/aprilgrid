from dataclasses import dataclass
from typing import List
import numpy as np


@dataclass
class Detection:
    tag_id: int
    corners: List[np.ndarray]
