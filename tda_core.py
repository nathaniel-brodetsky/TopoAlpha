import numpy as np
from ripser import ripser


class TDAAnalyzer:
    def __init__(self, maxdim=1):
        self.maxdim = maxdim

    def get_topological_stress(self, point_cloud):
        if len(point_cloud) < 10:
            return 0.0

        try:
            result = ripser(point_cloud, maxdim=self.maxdim)
            diagrams = result['dgms']

            h1 = diagrams[1]
            if len(h1) == 0:
                return 0.0

            valid_h1 = h1[h1[:, 1] != np.inf]
            if len(valid_h1) == 0:
                return 0.0

            persistences = valid_h1[:, 1] - valid_h1[:, 0]

            return float(np.max(persistences))
        except Exception:
            return 0.0
