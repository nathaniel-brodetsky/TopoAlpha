"""
topo_dyn_predictor.py  —  TopoAlpha Advanced Signal Engine
══════════════════════════════════════════════════════════════════════════════
Synthesises Topological Data Analysis (TDA) and Dynamical Systems Theory (DST)
for quantitative financial time-series analysis and directional prediction.

─── Bibliography ─────────────────────────────────────────────────────────────

TDA / Algebraic Topology
  Edelsbrunner & Harer (2010)   Computational Topology: An Introduction.
  Carlsson (2009)               Topology and Data. Bull. AMS 46(2):255-308.
  Bubenik (2015)                Statistical TDA Using Persistence Landscapes.
                                JMLR 16(1):77-102.
  Adams et al. (2017)           Persistence Images. JMLR 18(1):218-252.
  Gidea & Katz (2018)           TDA of Financial Time Series: Landscapes
                                of Crashes. Physica A 491:820-834.
  Perea & Harer (2015)          Sliding Windows and Persistence. Found.
                                Comput. Math. 15(3):799-838.
  Chintakunta et al. (2015)     Entropy-based Persistence Summaries.
                                Pattern Recognition 47(4):1469-1484.
  Rucco et al. (2016)           Characterisation via Persistent Entropy.
                                Complex Networks VII:117-128.

Dynamical Systems / Nonlinear Time-Series
  Takens (1981)                 Detecting Strange Attractors in Turbulence.
                                LNM 898:366-381.
  Packard et al. (1980)         Geometry from a Time Series. PRL 45:712.
  Kennel, Brown & Abarbanel (1992) Determining Embedding Dimension via FNN.
                                Phys. Rev. A 45(6):3403.
  Fraser & Swinney (1986)       Independent Coordinates via Mutual Information.
                                Phys. Rev. A 33(2):1134.
  Grassberger & Procaccia (1983) Measuring the Strangeness of Strange
                                Attractors. Physica D 9(1-2):189-208.
  Rosenstein, Collins & De Luca (1993) A Practical Method for Calculating
                                Largest Lyapunov Exponents. Physica D 65.
  Eckmann, Kamphorst & Ruelle (1987) Recurrence Plots of Dynamical Systems.
                                Europhys. Lett. 4(9):973.
  Zbilut & Webber (1992)        Embeddings and Delays from Recurrence Plots.
                                Phys. Lett. A 171(3-4):199-203.
  Kantz & Schreiber (2004)      Nonlinear Time Series Analysis. 2nd ed. CUP.
  Hurst (1951)                  Long-Term Storage Capacity of Reservoirs.
                                Trans. Am. Soc. Civil Eng. 116:770-799.
  Peng et al. (1994)            Mosaic Organisation of DNA Nucleotides (DFA).
                                Phys. Rev. E 49(2):1685.
  Kantelhardt et al. (2002)     Multifractal DFA. Physica A 316:87-114.

Information Theory / Complexity
  Bandt & Pompe (2002)          Permutation Entropy. PRL 88(17):174102.
  Richman & Moorman (2000)      Sample Entropy. Am. J. Physiol. 278(6):H2039.

Critical Transitions / Early Warning Signals
  Scheffer et al. (2009)        Early-Warning Signals. Nature 461:53-59.
  Dakos et al. (2008)           Slowing Down as Early Warning. PNAS 105:14308.

══════════════════════════════════════════════════════════════════════════════
"""
from __future__ import annotations

import logging
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FuturesTimeout
from dataclasses import dataclass, field
from math import factorial, log as mlog

import numpy as np
from scipy import stats
from scipy.spatial.distance import cdist

try:
    from ripser import ripser as _ripser
    _HAS_RIPSER = True
except ImportError:
    _HAS_RIPSER = False

logger = logging.getLogger("TopoAlpha.TopoDyn")


@dataclass
class TopoDynSignal:
    """
    Complete analysis output.

    direction   "LONG" | "SHORT" | "FLAT"
    confidence  Calibrated signal quality score in [0, 1].
    regime      "TRENDING" | "MEAN_REVERTING" | "CHAOTIC" | "TRANSITION" | "RANDOM"

    Core indicators
    ───────────────
    hurst       Ensemble Hurst exponent (R/S + DFA).
                >0.56 → persistent/trending.  <0.44 → anti-persistent.
    lyapunov    Maximal Lyapunov exponent (Rosenstein 1993).
                λ>0 → chaos/sensitivity.  λ<0 → stable quasi-periodic.
    perm_ent    Normalised permutation entropy (Bandt & Pompe 2002).
                0=ordered, 1=maximum complexity.
    corr_dim    Grassberger-Procaccia correlation dimension.
                Low (~1-2) → simple attractor.  High (>3) → complex dynamics.
    h1_stress   Maximum H₁ persistence (topological loop strength).
    h1_pers_ent Persistent entropy of H₁ diagram (information complexity).
    h1_velocity Rate of change of h1_stress between calls (crash precursor).
                Gidea & Katz (2018): rising stress precedes major reversals.
    h1_land_l1  L¹ norm of the H₁ persistence landscape (Bubenik 2015).
    rqa_det     Recurrence determinism. High → structured, predictable regime.
    ews_score   Critical-transition danger score in [0,1].
                >0.65 → approaching regime change (Scheffer 2009).
    mf_delta_h  Multifractal spectrum width (Kantelhardt 2002).
                High → strong non-linearity.
    features    Full feature dict for downstream ML pipelines.
    """
    direction   : str
    confidence  : float
    regime      : str

    hurst       : float
    lyapunov    : float
    perm_ent    : float
    sample_ent  : float
    corr_dim    : float
    h1_stress   : float
    h1_pers_ent : float
    h1_velocity : float
    h1_land_l1  : float
    rqa_det     : float
    rqa_entr    : float
    ews_score   : float
    mf_delta_h  : float

    features    : dict = field(default_factory=dict)

    def __str__(self) -> str:
        flag = {"LONG": "▲", "SHORT": "▼", "FLAT": "─"}.get(self.direction, "?")
        return (
            f"{flag} {self.direction:<5}  conf={self.confidence:.2f}  "
            f"[{self.regime}]  "
            f"H={self.hurst:.3f}  λ={self.lyapunov:+.4f}  "
            f"PE={self.perm_ent:.3f}  D={self.corr_dim:.2f}  "
            f"H1={self.h1_stress:.4f}  Δ={self.h1_velocity:+.4f}  "
            f"DET={self.rqa_det:.3f}  EWS={self.ews_score:.3f}  "
            f"Δh={self.mf_delta_h:.3f}"
        )



class TopoDynPredictor:
    """
    Advanced signal engine combining Topological Data Analysis and
    Dynamical Systems Theory.

    Design principles
    ─────────────────
    • Every computation gate is independent and fail-safe.
      A timeout or numerical failure returns a neutral default, so the
      signal degrades gracefully without crashing the trading loop.
    • Stateful velocity tracking: h1_velocity measures the change in
      topological stress between successive calls, capturing the
      Gidea-Katz crash precursor signal.
    • One ripser call per features() invocation — TDA is the bottleneck;
      all other computations are fast vectorised NumPy/SciPy.

    Parameters
    ──────────
    embed_dim       Phase-space embedding dimension (Takens theorem).
    max_tau         Upper bound for auto-selected time delay τ.
    tda_max_dim     Highest homology dimension (0=H₀, 1=H₁, 2=H₂).
    tda_cap         Max embedding points fed to ripser (speed cap).
    rqa_cap         Max embedding points for RQA distance matrix.
    rqa_eps_pct     Percentile of pairwise distances as recurrence threshold.
    ews_window      Lookback window for early-warning statistics.
    lya_steps       Divergence horizon for Lyapunov estimation.
    tda_timeout_s   Hard timeout for the ripser call.
    """

    def __init__(
        self,
        embed_dim     : int   = 3,
        max_tau       : int   = 20,
        tda_max_dim   : int   = 2,
        tda_cap       : int   = 120,
        rqa_cap       : int   = 100,
        rqa_eps_pct   : float = 10.0,
        ews_window    : int   = 40,
        lya_steps     : int   = 15,
        tda_timeout_s : float = 5.0,
    ) -> None:
        self.embed_dim     = embed_dim
        self.max_tau       = max_tau
        self.tda_max_dim   = tda_max_dim
        self.tda_cap       = tda_cap
        self.rqa_cap       = rqa_cap
        self.rqa_eps_pct   = rqa_eps_pct
        self.ews_window    = ews_window
        self.lya_steps     = lya_steps
        self.tda_timeout_s = tda_timeout_s
        self._tda_pool     = ThreadPoolExecutor(
            max_workers=1, thread_name_prefix="tda_worker"
        )
        self._prev_h1_stress : float = 0.0

    # ─────────────────────────────────────────────────────────────────────
    # I.  Phase-Space Reconstruction — Takens (1981)
    # ─────────────────────────────────────────────────────────────────────

    def _optimal_tau(self, x: np.ndarray) -> int:
        """
        Auto-select time delay τ as the first zero-crossing (or local minimum)
        of the autocorrelation function — a fast proxy for the mutual-information
        criterion of Fraser & Swinney (1986).
        """
        if len(x) < 4:
            return 1
        xc  = x - x.mean()
        std = xc.std()
        if std < 1e-12:
            return 1
        acf  = np.correlate(xc, xc, mode="full")[len(xc) - 1:]
        acf /= acf[0] + 1e-12
        for tau in range(1, min(self.max_tau, len(acf) - 1)):
            if acf[tau] <= 0.0:
                return tau
            if acf[tau - 1] > acf[tau] < acf[tau + 1]:
                return tau
        return min(5, self.max_tau)

    def _embed(self, x: np.ndarray, tau: int) -> np.ndarray:
        """Takens delay embedding: x → point cloud in ℝ^d."""
        d = self.embed_dim
        n = len(x) - (d - 1) * tau
        if n < 5:
            return np.empty((0, d))
        return np.column_stack([x[i * tau: i * tau + n] for i in range(d)])

    # ─────────────────────────────────────────────────────────────────────
    # II.  Persistent Homology — Edelsbrunner & Harer (2010)
    # ─────────────────────────────────────────────────────────────────────

    def _tda_features(self, cloud: np.ndarray) -> dict:
        """
        Vietoris-Rips persistent homology via ripser.

        Features extracted per dimension (H₀, H₁, H₂):
          _count      Number of finite persistence pairs.
          _stress     Maximum lifetime (max persistence).
          _pers_ent   Persistence entropy (Rucco 2016): H = -Σ pᵢ log₂ pᵢ.
          _sum_life   Total lifetime (sum of all lifetimes).
          _mean_life  Mean lifetime.
          _birth_mean Mean birth filtration value.
          _death_mean Mean death filtration value.

        Additional H₁ features:
          h1_land_l1  L¹ norm of persistence landscape λ₁ (Bubenik 2015).
          h1_land_l2  L² norm of persistence landscape λ₁.
        """
        zero = self._zero_tda()
        if not _HAS_RIPSER or len(cloud) < 10:
            return zero
        if cloud.std() < 1e-10:
            return zero

        cap     = min(len(cloud), self.tda_cap)
        cloud_c = cloud[-cap:]
        scale   = cloud_c.std() + 1e-10
        cloud_n = (cloud_c - cloud_c.mean(axis=0)) / scale

        try:
            future = self._tda_pool.submit(_ripser, cloud_n, maxdim=self.tda_max_dim)
            dgms   = future.result(timeout=self.tda_timeout_s)["dgms"]
        except (FuturesTimeout, Exception) as exc:
            logger.debug("[TDA] ripser: %s", exc)
            return zero

        out: dict = {}

        for dim in range(self.tda_max_dim + 1):
            pfx = f"h{dim}"
            _z  = {f"{pfx}_{k}": 0.0 for k in
                   ["count", "stress", "pers_ent", "sum_life",
                    "mean_life", "birth_mean", "death_mean"]}
            if dim >= len(dgms) or len(dgms[dim]) == 0:
                out.update(_z); continue

            dgm    = dgms[dim]
            finite = dgm[np.isfinite(dgm[:, 1])]
            if len(finite) == 0:
                out.update(_z); continue

            lives    = finite[:, 1] - finite[:, 0]
            total    = lives.sum() + 1e-12
            p_i      = lives / total
            pers_ent = float(-np.sum(p_i * np.log2(p_i + 1e-12)))

            out[f"{pfx}_count"]      = float(len(finite))
            out[f"{pfx}_stress"]     = float(lives.max())
            out[f"{pfx}_pers_ent"]   = pers_ent
            out[f"{pfx}_sum_life"]   = float(lives.sum())
            out[f"{pfx}_mean_life"]  = float(lives.mean())
            out[f"{pfx}_birth_mean"] = float(finite[:, 0].mean())
            out[f"{pfx}_death_mean"] = float(finite[:, 1].mean())

        # H₁ Persistence Landscape — Bubenik (2015)
        out["h1_land_l1"] = 0.0
        out["h1_land_l2"] = 0.0
        if len(dgms) > 1 and len(dgms[1]) > 0:
            fin1 = dgms[1][np.isfinite(dgms[1][:, 1])]
            if len(fin1) > 0:
                b_arr, d_arr = fin1[:, 0], fin1[:, 1]
                t_min, t_max = b_arr.min(), d_arr.max()
                if t_max > t_min:
                    tg  = np.linspace(t_min, t_max, 200)
                    lam = np.array([
                        max((min(t - b, d - t) for b, d in zip(b_arr, d_arr)
                             if b <= t <= d), default=0.0)
                        for t in tg
                    ])
                    out["h1_land_l1"] = float(np.trapz(np.abs(lam), tg))
                    out["h1_land_l2"] = float(np.sqrt(np.trapz(lam ** 2, tg)))

        return out

    @staticmethod
    def _zero_tda() -> dict:
        keys: list[str] = []
        for dim in range(3):
            k = f"h{dim}"
            keys += [f"{k}_count", f"{k}_stress", f"{k}_pers_ent",
                     f"{k}_sum_life", f"{k}_mean_life",
                     f"{k}_birth_mean", f"{k}_death_mean"]
        keys += ["h1_land_l1", "h1_land_l2"]
        return dict.fromkeys(keys, 0.0)

    # ─────────────────────────────────────────────────────────────────────
    # III.  Hurst Exponent — Hurst (1951) + Peng et al. (1994)
    # ─────────────────────────────────────────────────────────────────────

    def _hurst_rs(self, ret: np.ndarray) -> float:
        """Rescaled Range Analysis. Returns H ∈ [0,1]."""
        n = len(ret)
        if n < 20:
            return 0.5
        proportions = [0.125, 0.25, 0.375, 0.5, 0.625, 0.75]
        lags = [int(n * p) for p in proportions if int(n * p) >= 10]
        if len(lags) < 2:
            return 0.5
        rs_vals: list[float] = []
        for lag in lags:
            chunk = ret[:lag]
            dev   = np.cumsum(chunk - chunk.mean())
            S     = chunk.std(ddof=1)
            if S > 0:
                rs_vals.append((dev.max() - dev.min()) / S)
        if len(rs_vals) < 2:
            return 0.5
        slope, _ = np.polyfit(np.log(lags[:len(rs_vals)]), np.log(rs_vals), 1)
        return float(np.clip(slope, 0.0, 1.0))

    def _hurst_dfa(self, ret: np.ndarray) -> float:
        """Detrended Fluctuation Analysis. Returns H ∈ [0,1]."""
        n = len(ret)
        if n < 20:
            return 0.5
        y = np.cumsum(ret - ret.mean())
        scales = np.unique(
            np.logspace(np.log10(8), np.log10(max(9, n // 4)), 10).astype(int)
        )
        flucts      : list[float] = []
        valid_scales: list[int]   = []
        t_buf       = np.arange(max(scales) + 1)
        for s in scales:
            n_seg = n // s
            if n_seg < 2:
                continue
            tl   = t_buf[:s]
            rms2 : list[float] = []
            for k in range(n_seg):
                seg  = y[k * s: (k + 1) * s]
                coef = np.polyfit(tl, seg, 1)
                rms2.append(float(np.mean((seg - np.polyval(coef, tl)) ** 2)))
            if rms2:
                flucts.append(float(np.sqrt(np.mean(rms2))))
                valid_scales.append(int(s))
        if len(flucts) < 2:
            return 0.5
        slope, _ = np.polyfit(np.log(valid_scales), np.log(flucts), 1)
        return float(np.clip(slope, 0.0, 1.0))

    def _hurst(self, prices: np.ndarray) -> float:
        """Ensemble Hurst: mean of R/S and DFA estimates."""
        ret = np.diff(np.log(np.maximum(prices, 1e-10)))
        return float((self._hurst_rs(ret) + self._hurst_dfa(ret)) / 2.0)

    # ─────────────────────────────────────────────────────────────────────
    # IV.  Maximal Lyapunov Exponent — Rosenstein, Collins & De Luca (1993)
    # ─────────────────────────────────────────────────────────────────────

    def _lyapunov(self, cloud: np.ndarray) -> float:
        """
        Estimates the largest Lyapunov exponent λ₁ via the
        Rosenstein et al. divergence-tracking algorithm.

        λ₁ > 0  →  sensitive dependence on initial conditions (chaotic).
        λ₁ ≈ 0  →  marginally stable.
        λ₁ < 0  →  contracting, quasi-periodic (more predictable).
        """
        n = len(cloud)
        if n < 20:
            return 0.0

        D       = cdist(cloud, cloud)
        min_sep = max(1, n // 10)
        # Exclude temporal neighbourhood
        for i in range(n):
            lo = max(0, i - min_sep)
            hi = min(n, i + min_sep + 1)
            D[i, lo:hi] = np.inf
        np.fill_diagonal(D, np.inf)

        nn_idx = np.argmin(D, axis=1)

        log_div: list[float] = []
        for step in range(1, self.lya_steps + 1):
            i_valid = np.where(
                (np.arange(n) + step < n) & (nn_idx + step < n)
            )[0]
            if len(i_valid) == 0:
                break
            j_valid = nn_idx[i_valid]
            d0 = np.linalg.norm(cloud[i_valid]        - cloud[j_valid],        axis=1)
            dt = np.linalg.norm(cloud[i_valid + step] - cloud[j_valid + step], axis=1)
            good = d0 > 0
            if good.sum() == 0:
                continue
            log_div.append(float(np.mean(np.log(dt[good] / d0[good] + 1e-12))))

        if len(log_div) < 3:
            return 0.0
        t      = np.arange(len(log_div), dtype=float)
        lam, _ = np.polyfit(t, log_div, 1)
        return float(lam)

    # ─────────────────────────────────────────────────────────────────────
    # V.  Correlation Dimension — Grassberger & Procaccia (1983)
    # ─────────────────────────────────────────────────────────────────────

    def _correlation_dim(self, cloud: np.ndarray) -> float:
        """
        Estimates the fractal dimension of the reconstructed attractor.
        Slope of log C(r) vs log r where C(r) = fraction of pairs
        closer than r.

        Low D₂ (~1–2) → simple, low-dimensional dynamics.
        High D₂ (>4)  → complex, high-dimensional noise.
        """
        n = len(cloud)
        if n < 20:
            return 0.0
        D  = cdist(cloud, cloud)
        dv = D[np.triu_indices(n, k=1)]   # upper triangle, no diagonal
        if len(dv) == 0 or dv.std() < 1e-12:
            return 0.0
        r_pcts  = np.linspace(5, 50, 8)
        r_vals  = np.percentile(dv, r_pcts)
        C_r     = np.array([(dv < r).sum() / len(dv) for r in r_vals])
        valid   = C_r > 0
        if valid.sum() < 2:
            return 0.0
        slope, _ = np.polyfit(
            np.log(r_vals[valid]), np.log(C_r[valid]), 1
        )
        return float(np.clip(slope, 0.0, 10.0))

    # ─────────────────────────────────────────────────────────────────────
    # VI.  Recurrence Quantification Analysis — Eckmann (1987); Zbilut (1992)
    # ─────────────────────────────────────────────────────────────────────

    def _rqa(self, cloud: np.ndarray) -> dict:
        """
        Computes four standard RQA measures from the recurrence matrix
        R_{ij} = Θ(ε − ‖xᵢ − xⱼ‖):

          rqa_rr    Recurrence Rate:   fraction of recurrent states.
          rqa_det   Determinism:       fraction of points in diagonal lines ≥ 2.
                    High DET → deterministic, predictable structure.
          rqa_lmax  Longest diagonal line length — predictability horizon.
          rqa_entr  Shannon entropy of diagonal line-length distribution.
        """
        _z = {"rqa_rr": 0.0, "rqa_det": 0.0, "rqa_lmax": 0.0, "rqa_entr": 0.0}
        n  = len(cloud)
        if n < 10:
            return _z

        cap   = min(n, self.rqa_cap)
        cloud = cloud[-cap:]
        n     = len(cloud)

        D   = cdist(cloud, cloud)
        pos = D[D > 0]
        if len(pos) == 0:
            return _z
        eps = float(np.percentile(pos, self.rqa_eps_pct))
        R   = (D <= eps).astype(np.int8)
        np.fill_diagonal(R, 0)

        rr = float(R.sum()) / (n * (n - 1) + 1e-12)

        # Diagonal line lengths via numpy diagonal extraction
        min_l   = 2
        lengths : list[int] = []
        for k in range(-(n - 1), n):
            diag = np.diag(R, k)
            if len(diag) < min_l:
                continue
            pad  = np.concatenate([[0], diag.view(np.int8), [0]])
            diff = np.diff(pad.astype(int))
            ons  = np.where(diff == 1)[0]
            offs = np.where(diff == -1)[0]
            for a, b in zip(ons, offs):
                run = int(b - a)
                if run >= min_l:
                    lengths.append(run)

        if not lengths:
            return {"rqa_rr": rr, "rqa_det": 0.0, "rqa_lmax": 0.0, "rqa_entr": 0.0}

        la  = np.array(lengths)
        det = float(la.sum()) / (float(R.sum()) + 1e-12)
        lmx = float(la.max())

        counts = np.bincount(la)
        counts = counts[counts > 0]
        p      = counts / (counts.sum() + 1e-12)
        entr   = float(-np.sum(p * np.log2(p + 1e-12)))

        return {"rqa_rr": rr, "rqa_det": det, "rqa_lmax": lmx, "rqa_entr": entr}

    # ─────────────────────────────────────────────────────────────────────
    # VII.  Permutation Entropy — Bandt & Pompe (2002)
    # ─────────────────────────────────────────────────────────────────────

    def _perm_entropy(self, x: np.ndarray, order: int = 4, delay: int = 1) -> float:
        """
        Normalised permutation entropy in [0, 1].
        0 = perfectly ordered.  1 = maximum complexity (random).

        Captures the ordinal structure of the series with no
        distributional assumptions; robust to noise and outliers.
        """
        n = len(x)
        if n < order * delay + 1:
            return 1.0
        perms: dict = {}
        for i in range(n - (order - 1) * delay):
            pat = tuple(np.argsort(x[i: i + order * delay: delay], kind="stable"))
            perms[pat] = perms.get(pat, 0) + 1
        total  = sum(perms.values())
        p_arr  = np.array(list(perms.values()), dtype=float) / total
        pe     = float(-np.sum(p_arr * np.log2(p_arr + 1e-12)))
        max_pe = mlog(factorial(order), 2)
        return float(pe / max_pe) if max_pe > 0 else 0.0

    # ─────────────────────────────────────────────────────────────────────
    # VIII.  Sample Entropy — Richman & Moorman (2000)
    # ─────────────────────────────────────────────────────────────────────

    def _sample_entropy(self, x: np.ndarray, m: int = 2, r_factor: float = 0.2) -> float:
        """
        Conditional entropy of length-(m+1) templates given length-m matches.
        Lower SampEn → more regular / self-similar price dynamics.
        Capped at 60 bars (O(n²) computation).
        """
        cap = min(len(x), 60)
        xc  = x[-cap:]
        nc  = len(xc)
        if nc < 2 * m + 4:
            return 0.0
        r = r_factor * float(np.std(xc))
        if r == 0.0:
            return 0.0
        A, B = 0, 0
        for i in range(nc - m - 1):
            t_m  = xc[i: i + m]
            t_m1 = xc[i: i + m + 1]
            for j in range(i + 1, nc - m - 1):
                if np.max(np.abs(t_m - xc[j: j + m])) < r:
                    B += 1
                    if np.max(np.abs(t_m1 - xc[j: j + m + 1])) < r:
                        A += 1
        if B == 0:
            return 0.0
        return float(-np.log(max(A, 1) / (B + 1e-12)))

    # ─────────────────────────────────────────────────────────────────────
    # IX.  Multifractal DFA — Kantelhardt et al. (2002)
    # ─────────────────────────────────────────────────────────────────────

    def _mfdfa(self, ret: np.ndarray) -> dict:
        """
        Generalised Hurst exponents h(q) for q ∈ {-3,-1,0,1,3}.

        mf_delta_h   Δh = h(q_min) − h(q_max): singularity-spectrum width.
                     Large → strong multifractality / non-linear dynamics.
        mf_h_mean    Mean h(q) ≈ standard DFA Hurst over all q.
        mf_asymm     h(−3) − h(3): left-tail vs right-tail asymmetry.
                     Positive → large losses have longer memory than gains.
        """
        n = len(ret)
        if n < 30:
            return {"mf_delta_h": 0.0, "mf_h_mean": 0.5, "mf_asymm": 0.0}
        y      = np.cumsum(ret - ret.mean())
        q_vals = [-3.0, -1.0, 0.0, 1.0, 3.0]
        scales = np.unique(
            np.logspace(np.log10(8), np.log10(max(9, n // 4)), 8).astype(int)
        )
        h_q: list[float] = []
        for q in q_vals:
            flucts      : list[float] = []
            used_scales : list[int]   = []
            for s in scales:
                n_seg = n // s
                if n_seg < 2:
                    continue
                tl   = np.arange(s)
                rms2 : list[float] = []
                for k in range(n_seg):
                    seg  = y[k * s: (k + 1) * s]
                    coef = np.polyfit(tl, seg, 1)
                    rms2.append(float(np.mean((seg - np.polyval(coef, tl)) ** 2)))
                rms_arr = np.sqrt(np.maximum(rms2, 1e-20))
                fq = (
                    float(np.exp(0.5 * np.mean(np.log(rms_arr ** 2 + 1e-20))))
                    if q == 0.0
                    else float((np.mean(rms_arr ** q)) ** (1.0 / q))
                )
                flucts.append(fq)
                used_scales.append(int(s))
            if len(flucts) >= 2:
                slope, _ = np.polyfit(np.log(used_scales), np.log(flucts), 1)
                h_q.append(float(np.clip(slope, 0.0, 2.0)))
            else:
                h_q.append(0.5)

        h_arr = np.array(h_q)
        return {
            "mf_delta_h": float(h_arr.max() - h_arr.min()),
            "mf_h_mean" : float(h_arr.mean()),
            "mf_asymm"  : float(h_arr[0] - h_arr[-1]),
        }

    # ─────────────────────────────────────────────────────────────────────
    # X.  Critical-Transition Early Warning — Scheffer (2009); Dakos (2008)
    # ─────────────────────────────────────────────────────────────────────

    def _critical_ews(self, prices: np.ndarray) -> dict:
        """
        Three canonical early-warning signals (EWS) for critical slowing down:
          ews_var   Variance of the recent window.
          ews_ac1   Lag-1 autocorrelation (approaches 1 near bifurcation).
          ews_skew  Skewness (asymmetry builds up near tipping points).
          ews_kurt  Excess kurtosis (fat tails near transitions).
          ews_score Composite danger score in [0, 1].
                    >0.65 → elevated risk of regime change.
        """
        _z = {"ews_var": 0.0, "ews_ac1": 0.0, "ews_skew": 0.0,
              "ews_kurt": 0.0, "ews_score": 0.0}
        w = self.ews_window
        if len(prices) < w + 5:
            return _z

        now  = prices[-w:]
        prev = prices[-2 * w: -w] if len(prices) >= 2 * w else prices[:w]

        var_now  = float(np.var(now))
        var_prev = float(np.var(prev)) + 1e-12
        var_rise = float(np.clip((var_now / var_prev) - 1.0, -1.0, 3.0)) / 3.0

        try:
            ac1 = float(np.corrcoef(now[:-1], now[1:])[0, 1])
        except Exception:
            ac1 = 0.0
        ac1 = float(np.clip(ac1, -1.0, 1.0))

        skew = float(stats.skew(now))
        kurt = float(stats.kurtosis(now))

        ac1_score = float(np.clip(abs(ac1), 0.0, 1.0))
        ews_score = float(np.clip((var_rise + ac1_score) / 2.0, 0.0, 1.0))

        return {
            "ews_var"  : var_now,
            "ews_ac1"  : ac1,
            "ews_skew" : skew,
            "ews_kurt" : kurt,
            "ews_score": ews_score,
        }

    # ─────────────────────────────────────────────────────────────────────
    # XI.  Composite Directional Signal
    # ─────────────────────────────────────────────────────────────────────

    def _composite_signal(
        self,
        feats  : dict,
        prices : np.ndarray,
    ) -> tuple[str, float, str]:
        """
        Derives direction, calibrated confidence, and regime.

        Regime classification (priority order)
        ───────────────────────────────────────
        TRANSITION    ews_score > 0.65     (critical slowing down)
        CHAOTIC       λ₁ > 0.05           (positive Lyapunov exponent)
        TRENDING      H > 0.56            (persistent Hurst)
        MEAN_REVERTING H < 0.44           (anti-persistent Hurst)
        RANDOM        otherwise

        Directional signal
        ──────────────────
        Base signal: fast EMA vs slow EMA + 10-bar rate of change.
        Hurst scaling: amplify in trending, reverse in mean-reverting.
        Forced FLAT when confidence < 0.30 or regime is CHAOTIC/TRANSITION.

        Confidence contributions
        ────────────────────────
        +  rqa_det             predictable recurrence structure
        +  (1 − perm_ent)      low ordinal complexity
        +  |H − 0.5|           extreme Hurst → clearer regime
        −  ews_score           transition risk
        −  max(0, λ₁)          chaos penalty
        −  mf_delta_h          multifractal non-linearity
        −  h1_velocity (if +)  rising topological stress (crash precursor)
        """
        h       = float(feats.get("hurst",       0.5))
        lam     = float(feats.get("lyapunov",    0.0))
        pe      = float(feats.get("perm_ent",    0.5))
        det     = float(feats.get("rqa_det",     0.0))
        ews     = float(feats.get("ews_score",   0.0))
        h1_v    = float(feats.get("h1_velocity", 0.0))
        delta_h = float(feats.get("mf_delta_h",  0.0))

        if ews > 0.65:
            regime = "TRANSITION"
        elif lam > 0.05:
            regime = "CHAOTIC"
        elif h > 0.56:
            regime = "TRENDING"
        elif h < 0.44:
            regime = "MEAN_REVERTING"
        else:
            regime = "RANDOM"

        n    = len(prices)
        fast = max(1, int(n * 0.05))
        slow = max(2, int(n * 0.20))
        mom  = ((np.mean(prices[-fast:]) - np.mean(prices[-slow:])) /
                (np.mean(prices[-slow:]) + 1e-10))
        roc_w = min(10, n - 1)
        roc   = (prices[-1] - prices[-(roc_w + 1)]) / (prices[-(roc_w + 1)] + 1e-10)
        raw   = (mom + roc) / 2.0

        if regime == "TRENDING":
            adj = raw * (1.0 + 2.0 * (h - 0.5))
        elif regime == "MEAN_REVERTING":
            adj = -raw * (1.0 + 2.0 * (0.5 - h))
        elif regime in ("CHAOTIC", "TRANSITION"):
            adj = raw * 0.15
        else:
            adj = raw

        conf = 0.55
        conf += det            * 0.20
        conf += (1.0 - pe)     * 0.15
        conf += abs(h - 0.5)   * 0.20
        conf -= ews            * 0.25
        conf -= max(0.0, lam)  * 0.20
        conf -= delta_h        * 0.10
        conf -= np.clip(h1_v, 0.0, 0.2) / 0.2 * 0.15
        conf  = float(np.clip(conf, 0.05, 0.95))

        thresh = 0.00015
        if adj > thresh:
            direction = "LONG"
        elif adj < -thresh:
            direction = "SHORT"
        else:
            direction = "FLAT"

        if conf < 0.30 or regime in ("CHAOTIC", "TRANSITION"):
            direction = "FLAT"

        return direction, conf, regime


    def features(self, prices: np.ndarray, obi: float = 0.0) -> dict:
        """
        Compute the complete feature vector.

        Parameters
        ──────────
        prices   1-D numpy array of closing prices, most recent last.
                 Minimum 50 bars; quality improves significantly at 200+.
        obi      Order Book Imbalance ∈ [−1, 1] (optional; pass 0 if unavailable).

        Returns
        ───────
        dict with ~45 float features ready for downstream ML ingestion.
        Empty dict if prices < 50 bars.
        """
        prices = np.asarray(prices, dtype=float)
        if len(prices) < 50:
            logger.warning("[TopoDyn] Need ≥50 prices; got %d.", len(prices))
            return {}

        ret   = np.diff(np.log(np.maximum(prices, 1e-10)))
        tau   = self._optimal_tau(prices[-min(len(prices), 120):])
        cloud = self._embed(prices, tau)

        feats: dict = {}

        feats.update(self._tda_features(cloud))

        # Topological velocity — stateful, O(1); key signal from Gidea & Katz (2018)
        current_h1             = float(feats.get("h1_stress", 0.0))
        feats["h1_velocity"]   = current_h1 - self._prev_h1_stress
        self._prev_h1_stress   = current_h1

        feats["hurst"]      = self._hurst(prices)
        feats["lyapunov"]   = self._lyapunov(cloud) if len(cloud) <= 200 else 0.0
        feats["corr_dim"]   = self._correlation_dim(cloud)

        feats.update(self._rqa(cloud))

        feats["perm_ent"]   = self._perm_entropy(ret)
        feats["sample_ent"] = self._sample_entropy(ret)

        feats.update(self._mfdfa(ret))

        feats.update(self._critical_ews(prices))

        feats["embed_tau"] = float(tau)
        feats["obi"]       = float(obi)

        return feats

    def signal(self, prices: np.ndarray, obi: float = 0.0) -> TopoDynSignal:
        """
        Full analysis → TopoDynSignal.

        Parameters
        ──────────
        prices   1-D array of close prices (≥200 bars recommended).
        obi      Order Book Imbalance ∈ [−1, 1].

        Returns
        ───────
        TopoDynSignal with direction, confidence, regime, and all indicators.
        """
        feats = self.features(prices, obi)

        if not feats:
            return TopoDynSignal(
                direction="FLAT", confidence=0.0, regime="INSUFFICIENT_DATA",
                hurst=0.5, lyapunov=0.0, perm_ent=1.0, sample_ent=0.0,
                corr_dim=0.0, h1_stress=0.0, h1_pers_ent=0.0,
                h1_velocity=0.0, h1_land_l1=0.0, rqa_det=0.0,
                rqa_entr=0.0, ews_score=0.0, mf_delta_h=0.0, features={},
            )

        direction, conf, regime = self._composite_signal(feats, prices)

        return TopoDynSignal(
            direction   = direction,
            confidence  = conf,
            regime      = regime,
            hurst       = feats.get("hurst",       0.5),
            lyapunov    = feats.get("lyapunov",    0.0),
            perm_ent    = feats.get("perm_ent",    1.0),
            sample_ent  = feats.get("sample_ent",  0.0),
            corr_dim    = feats.get("corr_dim",    0.0),
            h1_stress   = feats.get("h1_stress",   0.0),
            h1_pers_ent = feats.get("h1_pers_ent", 0.0),
            h1_velocity = feats.get("h1_velocity", 0.0),
            h1_land_l1  = feats.get("h1_land_l1",  0.0),
            rqa_det     = feats.get("rqa_det",     0.0),
            rqa_entr    = feats.get("rqa_entr",    0.0),
            ews_score   = feats.get("ews_score",   0.0),
            mf_delta_h  = feats.get("mf_delta_h",  0.0),
            features    = feats,
        )