"""
Tests for gsd_puretime.rotor — the Sudjianto Spin(3) rotor extension.

Verifies:
  * Round-trip SO(3) ↔ Rotor3 to machine precision.
  * Rotor conjugation reproduces matrix multiplication.
  * Composition preserves unit norm.
  * Boundedness for free: rotor norms = 1 across 40 orders of
    magnitude in the input matrix scale.
"""
from __future__ import annotations

import numpy as np
import pytest

from gsd_puretime.rotor import (
    Rotor3,
    rotation_matrix_to_rotor,
    rotor_to_rotation_matrix,
    rotor_apply,
    rotor_compose,
    svd_to_rotor,
    rotor_emergent_rank,
)


def _random_so3(rng):
    """Draw a uniform sample from SO(3) via QR of a Gaussian matrix."""
    A = rng.standard_normal((3, 3))
    Q, _ = np.linalg.qr(A)
    if np.linalg.det(Q) < 0:
        Q[:, 0] = -Q[:, 0]
    return Q


@pytest.mark.parametrize("seed", [0, 1, 2, 42, 99])
def test_round_trip_SO3_rotor(seed):
    """SO(3) → Rotor3 → SO(3) returns the original matrix."""
    rng = np.random.default_rng(seed)
    R = _random_so3(rng)
    rotor = rotation_matrix_to_rotor(R)
    R_back = rotor_to_rotation_matrix(rotor)
    np.testing.assert_allclose(R, R_back, atol=1e-12)
    assert abs(rotor.norm - 1.0) < 1e-12


@pytest.mark.parametrize("seed", [0, 1, 2, 42, 99])
def test_rotor_conjugation_matches_matrix(seed):
    """Rotor conjugation R v R̃ equals matrix multiplication R · v."""
    rng = np.random.default_rng(seed)
    R = _random_so3(rng)
    rotor = rotation_matrix_to_rotor(R)
    v = rng.standard_normal(3)
    np.testing.assert_allclose(rotor_apply(rotor, v), R @ v, atol=1e-12)


def test_rotor_composition_preserves_norm():
    """Composition of unit rotors yields a unit rotor."""
    rng = np.random.default_rng(7)
    r1 = rotation_matrix_to_rotor(_random_so3(rng))
    r2 = rotation_matrix_to_rotor(_random_so3(rng))
    r12 = rotor_compose(r1, r2)
    assert abs(r12.norm - 1.0) < 1e-12


@pytest.mark.parametrize("scale", [1e-20, 1e-10, 1.0, 1e10, 1e20])
def test_boundedness_across_scales(scale):
    """Rotor norms stay at 1 across 40 orders of magnitude in input scale."""
    rng = np.random.default_rng(13)
    M = scale * rng.standard_normal((3, 3))
    rL, sigma, rR = svd_to_rotor(M)
    assert abs(rL.norm - 1.0) < 1e-12, f"left rotor norm drifted at scale {scale}"
    assert abs(rR.norm - 1.0) < 1e-12, f"right rotor norm drifted at scale {scale}"


def test_rank_diagnostic_consistency():
    """Rotor emergent rank matches classical singular-value rank diagnostic."""
    from gsd_puretime import ptmv, ec_matrix, emergent_rank
    rng = np.random.default_rng(11)
    T = 1500
    # 3-D panel with 1 cointegrating relation
    y2 = np.cumsum(rng.standard_normal(T))
    y3 = np.cumsum(rng.standard_normal(T))
    y1 = 0.6 * y2 + 0.5 * rng.standard_normal(T)   # cointegrated with y2
    Y = np.column_stack([y1, y2, y3])
    A, B = ptmv(Y); M = ec_matrix(A, B)
    r_classic, _ = emergent_rank(M, T=T, threshold=2.5)
    r_rotor, _, _, _ = rotor_emergent_rank(M, T=T, threshold=2.5)
    assert r_classic == r_rotor


def test_rank_deficient_input_gives_unit_rotors():
    """Even rank-1 matrices yield unit-norm rotors."""
    rng = np.random.default_rng(0)
    M = np.outer(rng.standard_normal(3), rng.standard_normal(3))
    rL, _, rR = svd_to_rotor(M)
    assert abs(rL.norm - 1.0) < 1e-10
    assert abs(rR.norm - 1.0) < 1e-10


def test_identity_input_gives_identity_rotor():
    """SVD of identity yields identity rotor."""
    rL, sigma, rR = svd_to_rotor(np.eye(3))
    assert abs(rL.s - 1.0) < 1e-12 and np.linalg.norm(rL.b) < 1e-12
    assert abs(rR.s - 1.0) < 1e-12 and np.linalg.norm(rR.b) < 1e-12
    np.testing.assert_allclose(sigma, [1.0, 1.0, 1.0])
