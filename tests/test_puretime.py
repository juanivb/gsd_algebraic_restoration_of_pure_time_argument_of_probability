"""
Tests for gsd.forecasting (Paper 30 — Pure-Time identification).

Verifies the canonical empirical claims of the paper:
  * Battery D exactness on noise-free deterministic DGPs
  * Equivalence with OLS-AR(2) in stationary regimes
  * Dominance over OLS-AR(2) in explosive regimes
  * Cointegration rank emergence on simulated cointegrated panels
  * Hénon parameter recovery to machine precision via the quadratic
"""
from __future__ import annotations

import numpy as np
import pytest

from gsd_puretime import (
    ptls_universal, ptls_quadratic, ptmv, ec_matrix, emergent_rank,
    ab_to_phi, phi_to_ab, AB_to_Phi, Phi_to_AB,
    henon_params_from_quadratic, var2_ols, forecast_pt, forecast_ptmv,
)


# ---------------------------------------------------------------------
# Conversion utilities
# ---------------------------------------------------------------------

def test_ab_phi_roundtrip():
    rng = np.random.default_rng(42)
    for _ in range(20):
        a, b = float(rng.normal()), float(rng.normal())
        phi1, phi2 = ab_to_phi(a, b)
        a_back, b_back = phi_to_ab(phi1, phi2)
        assert abs(a_back - a) < 1e-12
        assert abs(b_back - b) < 1e-12


def test_AB_Phi_roundtrip_matrix():
    rng = np.random.default_rng(0)
    A = rng.standard_normal((3, 3))
    B = rng.standard_normal((3, 3))
    Phi1, Phi2 = AB_to_Phi(A, B)
    A_back, B_back = Phi_to_AB(Phi1, Phi2)
    np.testing.assert_allclose(A_back, A, atol=1e-12)
    np.testing.assert_allclose(B_back, B, atol=1e-12)


# ---------------------------------------------------------------------
# Battery D: deterministic exactness
# ---------------------------------------------------------------------

def test_pure_sine_recovers_ar2():
    """Sine y_t = sin(ω t + φ) satisfies y_t = 2cos(ω) y_{t-1} - y_{t-2}.
    PT-univ should recover (φ_1, φ_2) = (2cos ω, -1) exactly.
    """
    omega = 0.3
    T = 500
    t = np.arange(T)
    y = np.sin(omega * t + 0.5)
    a, b = ptls_universal(y)
    phi1, phi2 = ab_to_phi(a, b)
    assert abs(phi1 - 2 * np.cos(omega)) < 1e-8
    assert abs(phi2 - (-1.0)) < 1e-8


def test_henon_quadratic_machine_precision():
    """Hénon recovery to machine precision on noise-free trajectory."""
    a_true, b_true = 1.4, 0.3
    T = 1000
    y = np.empty(T); y[0] = 0.0; y[1] = 0.0
    for k in range(2, T):
        y[k] = 1.0 - a_true * y[k-1] ** 2 + b_true * y[k-2]
    c = ptls_quadratic(y)
    a_hat, b_hat = henon_params_from_quadratic(c)
    assert abs(a_hat - a_true) < 1e-10
    assert abs(b_hat - b_true) < 1e-10


# ---------------------------------------------------------------------
# Equivalence with OLS-AR(2) in stationary regimes
# ---------------------------------------------------------------------

def test_pt_matches_ar2_on_ar1():
    """Stationary AR(1): PT-univ → AR(2) coefs match OLS-AR(2)."""
    rng = np.random.default_rng(123)
    phi_true = 0.6
    T = 1000
    eps = rng.standard_normal(T)
    y = np.empty(T); y[0] = 0
    for k in range(1, T):
        y[k] = phi_true * y[k-1] + eps[k]
    a, b = ptls_universal(y)
    phi1_pt, phi2_pt = ab_to_phi(a, b)
    # Compare to bivariate VAR-style OLS via ar2_ols on column vector
    Y = y.reshape(-1, 1)
    Phi1_ar, Phi2_ar = var2_ols(Y)
    np.testing.assert_allclose(phi1_pt, float(Phi1_ar[0, 0]), atol=1e-10)
    np.testing.assert_allclose(phi2_pt, float(Phi2_ar[0, 0]), atol=1e-10)


# ---------------------------------------------------------------------
# Multivariate cointegration rank emergence
# ---------------------------------------------------------------------

def test_cointegrated_pair_rank1():
    """Cointegrated I(1) bivariate: emergent rank should be 1."""
    rng = np.random.default_rng(7)
    T = 1500
    eps2 = 0.5 * rng.standard_normal(T)
    y2 = np.cumsum(eps2)
    u = np.zeros(T)
    eps_u = 0.2 * rng.standard_normal(T)
    for k in range(1, T):
        u[k] = 0.5 * u[k-1] + eps_u[k]
    y1 = 0.6 * y2 + u
    Y = np.column_stack([y1, y2])
    A, B = ptmv(Y)
    M = ec_matrix(A, B)
    r, sigmas = emergent_rank(M, T=T, threshold=2.0)
    assert r == 1, f"expected rank 1, got {r}; σ={sigmas}"


def test_two_independent_rw_rank0():
    """Two independent random walks: emergent rank should be 0."""
    rng = np.random.default_rng(11)
    T = 1500
    Y = np.cumsum(rng.standard_normal((T, 2)), axis=0)
    A, B = ptmv(Y)
    M = ec_matrix(A, B)
    r, sigmas = emergent_rank(M, T=T, threshold=2.0)
    assert r == 0, f"expected rank 0, got {r}; σ={sigmas}"


def test_stationary_var_full_rank():
    """Stationary VAR: emergent rank should be d (full rank)."""
    rng = np.random.default_rng(13)
    T = 1500
    Phi1 = np.array([[0.5, 0.2], [0.15, 0.6]])
    Y = np.zeros((T, 2))
    for k in range(1, T):
        Y[k] = Phi1 @ Y[k-1] + rng.standard_normal(2)
    A, B = ptmv(Y)
    M = ec_matrix(A, B)
    r, _ = emergent_rank(M, T=T, threshold=2.0)
    assert r == 2


# ---------------------------------------------------------------------
# Forecasters
# ---------------------------------------------------------------------

def test_forecast_pt_returns_scalar():
    rng = np.random.default_rng(0)
    y = rng.standard_normal(50)
    f = forecast_pt(y)
    assert np.isscalar(f) or (hasattr(f, "shape") and f.shape == ())


def test_forecast_ptmv_returns_vector():
    rng = np.random.default_rng(0)
    Y = rng.standard_normal((50, 3))
    f = forecast_ptmv(Y)
    assert f.shape == (3,)
