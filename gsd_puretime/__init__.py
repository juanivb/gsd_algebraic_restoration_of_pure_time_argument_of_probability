"""
gsd.forecasting — Pure-Time Identification and Forecasting [Paper 30]

Implements the canonical pure-time estimators developed in Paper 30 of
the GSD programme. The submodule provides:

  * :func:`ptls_universal` — univariate level-1 forward equation estimator
    ``y_{t+1} = a y_t + b Δy_t + ε_{t+1}``, valid across all linear
    regimes (stationary, integrated, explosive, trending).

  * :func:`ptls_quadratic` — quadratic extension for bounded chaos
    (logistic, Hénon, etc.). Six features in the pure-time basis.

  * :func:`ptmv` — multivariate generalisation for vector processes
    ``y_{t+1} = A y_t + B Δy_t + ε_{t+1}``, with cointegration rank
    emerging from the SVD of ``Π = A + B - I``.

  * :func:`emergent_rank` — Johansen-style cointegration rank
    diagnostic without prior specification of ``r``.

  * :func:`forecast_pt`, :func:`forecast_pt_quad`, :func:`forecast_ptmv`
    — one-step-ahead recursive forecasters built on the estimators.

  * Conversion utilities: :func:`ab_to_phi`, :func:`phi_to_ab`,
    :func:`AB_to_Phi`, :func:`Phi_to_AB`.

  * :func:`var2_ols` and :func:`vecm_eg_2step` are kept as classical
    baselines for benchmarking.

For background and full empirical validation see the paper. The
relationship between this submodule's primitives and the classical
literature is articulated in §8 of the paper:

* OLS-AR(p) is the special case of :func:`ptls_universal` in the
  basis :math:`(y_{t-1}, y_{t-2})`; the pure-time basis
  :math:`(y_t, \\Delta y_t)` carries the same algebraic content but
  is well-conditioned in regimes where Cartesian regressors are not.

* Yule-Walker assumes stationarity; :func:`ptls_universal` does not.

* VECM (Johansen) requires the cointegration rank to be specified;
  :func:`ptmv` returns it as a by-product of unconstrained OLS.

References
----------
* Hamilton 1834, *Algebra as the Science of Pure Time* — origin of the
  triplet-of-steps reading.
* Phillips 1987, *Time series regression with a unit root* — modern
  asymptotic theory for non-stationary regression.
* Johansen 1991, *Estimation and hypothesis testing of cointegration* —
  classical cointegration rank test.
* Paper 7 (GSD) — Geometric Takens theorem, dimensional invariant
  ``d†``.
* Paper 24 (GSD) — Moment-tensor reading.
* Paper 28 (GSD) — Non-abelian characteristic function.
* Paper 30 (GSD) — This paper.
"""
from __future__ import annotations

from typing import Tuple, Optional

import numpy as np


__all__ = [
    "ptls_universal",
    "ptls_quadratic",
    "ptmv",
    "emergent_rank",
    "ec_matrix",
    "ab_to_phi", "phi_to_ab",
    "AB_to_Phi", "Phi_to_AB",
    "henon_params_from_quadratic",
    "logistic_param_from_quadratic",
    "forecast_pt", "forecast_pt_quad", "forecast_ptmv",
    "var2_ols", "vecm_eg_2step",
]


# =====================================================================
# Univariate
# =====================================================================

def ptls_universal(y: np.ndarray) -> Tuple[float, float]:
    """Pure-time level-1 universal estimator (univariate).

    Fits ``y_{t+1} = a y_t + b Δy_t + ε_{t+1}`` by OLS.

    Parameters
    ----------
    y : array, shape (T,)
        Univariate series.

    Returns
    -------
    (a, b) : tuple of floats
        Estimated coefficients. Convert to AR(2) representation via
        :func:`ab_to_phi`.

    Notes
    -----
    Regime-agnostic: consistent under stationary AR(p≤2), I(1) random
    walk, I(2), explosive AR, and polynomial trends. See
    :func:`emergent_rank` for the multivariate cointegration version.
    """
    y = np.asarray(y, dtype=float).ravel()
    if len(y) < 4:
        return 0.0, 0.0
    y_t = y[1:-1]
    dy_t = y[1:-1] - y[:-2]
    y_next = y[2:]
    X = np.column_stack([y_t, dy_t])
    coef, *_ = np.linalg.lstsq(X, y_next, rcond=None)
    return float(coef[0]), float(coef[1])


def ptls_quadratic(y: np.ndarray) -> np.ndarray:
    """Pure-time quadratic estimator for bounded nonlinear systems.

    Fits

        ``y_{t+1} = c_0 + c_1 y_t + c_2 Δy_t + c_3 y_t^2 + c_4 (Δy_t)^2 + c_5 y_t·Δy_t + ε``

    Returns the 6-coordinate coefficient vector ``(c_0, c_1, c_2, c_3, c_4, c_5)``.

    Notes
    -----
    Recovers Hénon ``y_{t+1} = 1 - a y_t² + b y_{t-1}`` to machine
    precision on noise-free data (recovers ``a = -c_3``, ``b = c_1``).
    For one-dimensional chaos (logistic), the 6-feature design matrix
    is rank-5 and the natural reduced basis ``(1, y_t, y_t^2)`` should
    be used instead; see :func:`logistic_param_from_quadratic`.

    UNSTABLE under explosive extrapolation: the term ``y_t^2`` grows
    as ``φ^{2t}``, so the quadratic estimator is for bounded systems
    only.
    """
    y = np.asarray(y, dtype=float).ravel()
    if len(y) < 4:
        return np.zeros(6)
    y_t = y[1:-1]
    dy_t = y[1:-1] - y[:-2]
    y_next = y[2:]
    X = np.column_stack([
        np.ones_like(y_t),
        y_t,
        dy_t,
        y_t * y_t,
        dy_t * dy_t,
        y_t * dy_t,
    ])
    coef, *_ = np.linalg.lstsq(X, y_next, rcond=None)
    return coef


# =====================================================================
# Multivariate
# =====================================================================

def ptmv(Y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Pure-time multivariate level-1 estimator.

    Fits ``y_{t+1} = A y_t + B Δy_t + ε_{t+1}`` for matrices
    ``A, B ∈ R^{d×d}``.

    Parameters
    ----------
    Y : array, shape (T, d)
        Multivariate series.

    Returns
    -------
    (A, B) : tuple of arrays, each (d, d)
        Estimated coefficient matrices. Convert to VAR(2) representation
        via :func:`AB_to_Phi`. The error-correction matrix
        ``Π = A + B - I`` (see :func:`ec_matrix`) admits a Johansen-style
        emergent rank diagnostic via :func:`emergent_rank`.
    """
    Y = np.asarray(Y, dtype=float)
    if Y.ndim == 1:
        Y = Y.reshape(-1, 1)
    T, d = Y.shape
    if T < 4:
        return np.eye(d), np.zeros((d, d))
    Y_t = Y[1:-1]
    dY_t = Y[1:-1] - Y[:-2]
    Y_next = Y[2:]
    X = np.hstack([Y_t, dY_t])
    coef, *_ = np.linalg.lstsq(X, Y_next, rcond=None)
    A = coef[:d].T
    B = coef[d:].T
    return A, B


def ec_matrix(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    """Error-correction matrix ``Π = A + B - I``.

    The cointegration rank of the underlying process equals the rank
    of ``Π``. Use :func:`emergent_rank` for a finite-sample diagnostic.
    """
    d = A.shape[0]
    return A + B - np.eye(d)


def emergent_rank(M: np.ndarray, T: int,
                     threshold: float = 2.0) -> Tuple[int, np.ndarray]:
    """Johansen-style emergent cointegration rank.

    Parameters
    ----------
    M : array, shape (d, d)
        The error-correction matrix ``Π̂`` (e.g.\ from :func:`ec_matrix`).
    T : int
        Sample size used to fit ``M``.
    threshold : float, default 2.0
        Cutoff for ``σ_i √T``. Singular values with ``σ_i √T > threshold``
        are declared effective; smaller values are deemed spurious
        (data noise). The default is conservative; tune for the
        application.

    Returns
    -------
    rank : int
        Number of effective singular directions, equivalently the
        cointegration rank ``r``.
    sigmas : array, shape (d,)
        Singular values, decreasing.

    Notes
    -----
    Under the cointegration null with rank ``r``, the smallest
    ``d - r`` singular values of ``Π̂`` are :math:`O_p(T^{-1/2})`, while
    the top ``r`` are :math:`O_p(1)`. The diagnostic
    ``σ_i √T > threshold`` separates these asymptotic orders cleanly
    in samples ``T ≳ 100``.
    """
    s = np.linalg.svd(M, compute_uv=False)
    sqrtT = float(np.sqrt(T))
    eff = int(np.sum(s * sqrtT > threshold))
    return eff, s


# =====================================================================
# Conversions: pure-time ↔ classical
# =====================================================================

def ab_to_phi(a: float, b: float) -> Tuple[float, float]:
    """Convert pure-time ``(a, b)`` to AR(2) ``(φ_1, φ_2)``.

    From the forward equation ``y_{t+1} = a y_t + b Δy_t``:
    ``φ_1 = a + b``, ``φ_2 = -b``.
    """
    return a + b, -b


def phi_to_ab(phi1: float, phi2: float) -> Tuple[float, float]:
    """Convert AR(2) ``(φ_1, φ_2)`` to pure-time ``(a, b)``.

    Inverse of :func:`ab_to_phi`: ``a = φ_1 + φ_2``, ``b = -φ_2``.
    """
    return phi1 + phi2, -phi2


def AB_to_Phi(A: np.ndarray, B: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Convert pure-time ``(A, B)`` to VAR(2) ``(Φ_1, Φ_2)``.

    ``Φ_1 = A + B``, ``Φ_2 = -B``.
    """
    return A + B, -B


def Phi_to_AB(Phi1: np.ndarray, Phi2: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Convert VAR(2) ``(Φ_1, Φ_2)`` to pure-time ``(A, B)``.

    Inverse of :func:`AB_to_Phi`: ``A = Φ_1 + Φ_2``, ``B = -Φ_2``.
    """
    return Phi1 + Phi2, -Phi2


def henon_params_from_quadratic(c: np.ndarray) -> Tuple[float, float]:
    """Recover Hénon ``(a, b)`` from the quadratic coefficient vector.

    Hénon ``y_{t+1} = 1 - a y_t² + b y_{t-1}`` corresponds to
    ``c = (1, b, -b, -a, 0, 0)`` in the quadratic basis of
    :func:`ptls_quadratic`.
    """
    return float(-c[3]), float(c[1])


def logistic_param_from_quadratic(c: np.ndarray) -> float:
    """Recover logistic ``r`` from the quadratic coefficient vector.

    Logistic ``y_{t+1} = r y_t (1 - y_t)`` corresponds to
    ``c = (0, r, 0, -r, 0, 0)`` in the natural reduced basis. Averages
    the two redundant estimates for stability.
    """
    return float(0.5 * (c[1] - c[3]))


# =====================================================================
# Forecasters
# =====================================================================

def forecast_pt(y_history: np.ndarray, a: Optional[float] = None,
                  b: Optional[float] = None) -> float:
    """One-step-ahead forecast under the pure-time forward equation.

    If ``a, b`` are ``None``, fits them on ``y_history`` first via
    :func:`ptls_universal`.
    """
    y_history = np.asarray(y_history, dtype=float).ravel()
    if a is None or b is None:
        a, b = ptls_universal(y_history)
    if len(y_history) < 2:
        return float(a) * float(y_history[-1])
    y_t = y_history[-1]
    dy_t = y_history[-1] - y_history[-2]
    return float(a) * y_t + float(b) * dy_t


def forecast_pt_quad(y_history: np.ndarray,
                       c: Optional[np.ndarray] = None) -> float:
    """One-step forecast under the quadratic pure-time model.

    For bounded nonlinear systems only. Will diverge if the underlying
    process is explosive.
    """
    y_history = np.asarray(y_history, dtype=float).ravel()
    if c is None:
        c = ptls_quadratic(y_history)
    if len(y_history) < 2:
        return float(c[0])
    y_t = y_history[-1]
    dy_t = y_history[-1] - y_history[-2]
    feat = np.array([1.0, y_t, dy_t, y_t**2, dy_t**2, y_t * dy_t])
    return float(c @ feat)


def forecast_ptmv(Y_history: np.ndarray, A: Optional[np.ndarray] = None,
                    B: Optional[np.ndarray] = None) -> np.ndarray:
    """Multivariate one-step pure-time forecast."""
    Y_history = np.asarray(Y_history, dtype=float)
    if A is None or B is None:
        A, B = ptmv(Y_history)
    if Y_history.shape[0] < 2:
        return A @ Y_history[-1]
    y_t = Y_history[-1]
    dy_t = Y_history[-1] - Y_history[-2]
    return A @ y_t + B @ dy_t


# =====================================================================
# Classical baselines (for benchmarking)
# =====================================================================

def var2_ols(Y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Standard VAR(2) OLS: ``y_t = Φ_1 y_{t-1} + Φ_2 y_{t-2} + ε``.

    Provided as a convenience baseline. Use :func:`ptmv` instead for
    regime-agnostic estimation.
    """
    Y = np.asarray(Y, dtype=float)
    if Y.ndim == 1:
        Y = Y.reshape(-1, 1)
    T, d = Y.shape
    if T < 4:
        return np.eye(d), np.zeros((d, d))
    Y_target = Y[2:]
    X = np.hstack([Y[1:-1], Y[:-2]])
    coef, *_ = np.linalg.lstsq(X, Y_target, rcond=None)
    return coef[:d].T, coef[d:].T


def vecm_eg_2step(Y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Engle-Granger 2-step VECM (bivariate-only, classical baseline).

    Returns ``(Π, Γ_1)`` where ``Π = α β^T`` is rank-1 by construction.
    """
    Y = np.asarray(Y, dtype=float)
    if Y.ndim != 2 or Y.shape[1] != 2:
        d = Y.shape[1] if Y.ndim == 2 else 1
        return np.zeros((d, d)), np.zeros((d, d))
    T = Y.shape[0]
    if T < 5:
        return np.zeros((2, 2)), np.zeros((2, 2))
    y1 = Y[:, 0]; y2 = Y[:, 1]
    X1 = np.column_stack([np.ones(T), y2])
    coef1, *_ = np.linalg.lstsq(X1, y1, rcond=None)
    beta_const, beta_21 = coef1
    u = y1 - beta_const - beta_21 * y2
    dY = np.diff(Y, axis=0)
    u_lag = u[:-2]
    dY_lag = dY[:-1]
    target = dY[1:]
    X2 = np.column_stack([u_lag, dY_lag])
    coef2, *_ = np.linalg.lstsq(X2, target, rcond=None)
    alpha = coef2[0]
    Gamma1 = coef2[1:].T
    Pi = np.outer(alpha, np.array([1.0, -beta_21]))
    return Pi, Gamma1
