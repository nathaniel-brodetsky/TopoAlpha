import numpy as np
from concurrent.futures import ThreadPoolExecutor, Future, TimeoutError as FuturesTimeoutError

from ripser import ripser

_TDA_EXECUTOR = ThreadPoolExecutor(max_workers=1, thread_name_prefix="tda")

RIPSER_TIMEOUT_S: float = 4.0
_DEGENERATE_STD_THRESHOLD: float = 1e-6


class TDAAnalyzer:
    """Topological stress via H1 persistence of a 3-D delay-embedded point cloud.

    Stress = max persistence of loop (H₁) features in the Vietoris–Rips
    filtration.  High values indicate strong cyclical structure in recent price
    trajectories — the core entry signal for TopoAlpha.
    """

    def __init__(self, maxdim: int = 1):
        self.maxdim = maxdim
        self._pending_future: Future | None = None

    def _compute(self, point_cloud: np.ndarray) -> float:
        diagrams = ripser(point_cloud, maxdim=self.maxdim)["dgms"]
        h1 = diagrams[1]
        if len(h1) == 0:
            return 0.0
        finite_h1 = h1[h1[:, 1] != np.inf]
        if len(finite_h1) == 0:
            return 0.0
        return float(np.max(finite_h1[:, 1] - finite_h1[:, 0]))

    def get_topological_stress(self, point_cloud: np.ndarray) -> float:
        if len(point_cloud) < 10:
            return 0.0
        if float(np.std(point_cloud)) < _DEGENERATE_STD_THRESHOLD:
            return 0.0
        if self._pending_future is not None and not self._pending_future.done():
            return 0.0
        try:
            self._pending_future = _TDA_EXECUTOR.submit(self._compute, point_cloud)
            return self._pending_future.result(timeout=RIPSER_TIMEOUT_S)
        except FuturesTimeoutError:
            return 0.0
        except Exception:
            return 0.0