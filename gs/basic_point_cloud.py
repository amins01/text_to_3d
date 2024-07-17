import numpy as np
from typing import NamedTuple

class BasicPointCloud(NamedTuple):
    points: np.array
    colors: np.array
    normals: np.array