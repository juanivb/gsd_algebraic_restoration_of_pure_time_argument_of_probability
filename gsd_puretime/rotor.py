"""
gsd_puretime.rotor — rotor representation in Spin(3) ⊂ Cl(3,0).

Implements the Sudjianto trick: convert the SVD of a matrix M ∈ R^{3×3}
into a (rotor R ∈ Spin(3), singular spectrum σ ∈ R^3_+) pair, where R
is the unit-norm rotor encoding the rotation U V^T of the SVD and σ
is the diagonal of Σ.

Why this matters for the programme:
- Spin(3) is a compact Lie group, so ‖R‖ = 1 by construction.
- The rotor representation gives stationarity (boundedness) for free
  on the rotation part of any linear estimator. The numerical blowup
  that the quadratic extension suffers under explosive extrapolation
  cannot occur in Spin(3).
- The construction extends literally to Spin(d) for d ≥ 4 with the
  same compactness guarantee, lifting the d=3 ceiling of the
  Geometric Takens theorem in computation while preserving it in
  identification.

Conventions:
- Bivector basis (e_{12}, e_{13}, e_{23}) ↔ axial vector via the
  Hodge dual mapping (e_{23}, -e_{13}, e_{12}) ↔ (e_1, e_2, e_3).
- A rotor R = (s, B) is stored as scalar s and bivector
  B = (b_{12}, b_{13}, b_{23}).
- Rotor acts on a vector v as v' = R v R^{-1} = R v R̃ where R̃ is the
  reversion (s, -B).
"""
from __future__ import annotations

from typing import Tuple, NamedTuple

import numpy as np


class Rotor3(NamedTuple):
    """A rotor in Spin(3) ⊂ Cl(3,0).

    Stored as scalar s and bivector b = (b12, b13, b23).
    Unit-norm by construction: s² + b12² + b13² + b23² = 1.
    """
    s: float
    b: np.ndarray  # shape (3,) — components in basis (e12, e13, e23)

    @property
    def norm_squared(self) -> float:
        return float(self.s * self.s + self.b @ self.b)

    @property
    def norm(self) -> float:
        return float(np.sqrt(self.norm_squared))

    def reverse(self) -> "Rotor3":
        """Reversion R̃ = s − B."""
        return Rotor3(self.s, -self.b)

    def __repr__(self) -> str:
        return (f"Rotor3(s={self.s:+.6f}, "
                f"b12={self.b[0]:+.6f}, b13={self.b[1]:+.6f}, "
                f"b23={self.b[2]:+.6f}, ‖R‖={self.norm:.6f})")


# ---------------------------------------------------------------------
# Core: SO(3) rotation matrix → Rotor3
# ---------------------------------------------------------------------

def rotation_matrix_to_rotor(R: np.ndarray, eps: float = 1e-12) -> Rotor3:
    """Convert R ∈ SO(3) to its rotor in Spin(3).

    Uses the standard axis-angle decomposition. For R = exp(θ K) with
    K = [[0,-n3,n2],[n3,0,-n1],[-n2,n1,0]] the skew matrix of axis
    n = (n1, n2, n3), the rotor is
        R̂ = cos(θ/2) − sin(θ/2)(n1 e_{23} − n2 e_{13} + n3 e_{12}).
    The minus sign in front of sin(θ/2) makes the conjugation rotate
    vectors in the same sense as R (right-handed convention).
    """
    R = np.asarray(R, dtype=float)
    if R.shape != (3, 3):
        raise ValueError(f"R must be 3x3, got {R.shape}")

    # Symmetrise away from the SO(3) manifold by snapping det to ±1.
    det = float(np.linalg.det(R))
    if det < 0:
        # Improper rotation; reflect by flipping one column of the
        # underlying SVD. Caller must handle if intent was orientation.
        raise ValueError(f"R has determinant {det:.4f} < 0; must be in SO(3).")

    cos_theta = (np.trace(R) - 1.0) / 2.0
    cos_theta = float(np.clip(cos_theta, -1.0, 1.0))
    theta = np.arccos(cos_theta)

    if theta < eps:
        # Near identity: rotor is (1, 0).
        return Rotor3(1.0, np.zeros(3))

    if np.pi - theta < eps:
        # Near π rotation: extract axis from the largest diagonal of (R + I)/2.
        M = (R + np.eye(3)) / 2.0
        diag = np.diag(M)
        i = int(np.argmax(diag))
        axis = M[:, i] / max(np.sqrt(diag[i]), eps)
        n1, n2, n3 = axis
    else:
        sin_theta = np.sin(theta)
        n1 = (R[2, 1] - R[1, 2]) / (2.0 * sin_theta)
        n2 = (R[0, 2] - R[2, 0]) / (2.0 * sin_theta)
        n3 = (R[1, 0] - R[0, 1]) / (2.0 * sin_theta)

    half = theta / 2.0
    s = float(np.cos(half))
    sin_half = float(np.sin(half))
    # Axial vector n ↔ bivector (n3 e_12, -n2 e_13, n1 e_23) (Hodge dual)
    b12 = -sin_half * n3
    b13 = +sin_half * n2
    b23 = -sin_half * n1
    return Rotor3(s, np.array([b12, b13, b23]))


def rotor_to_rotation_matrix(R: Rotor3) -> np.ndarray:
    """Inverse of rotation_matrix_to_rotor: rebuild R ∈ SO(3) from rotor."""
    s = R.s
    b12, b13, b23 = R.b
    # Axis (Hodge dual of bivector) and angle from rotor norm split.
    sin_half = float(np.sqrt(b12 * b12 + b13 * b13 + b23 * b23))
    if sin_half < 1e-12:
        return np.eye(3)
    n1 = -b23 / sin_half
    n2 = +b13 / sin_half
    n3 = -b12 / sin_half
    cos_half = s
    theta = 2.0 * np.arctan2(sin_half, cos_half)
    K = np.array([[0, -n3, n2], [n3, 0, -n1], [-n2, n1, 0]])
    return np.eye(3) + np.sin(theta) * K + (1 - np.cos(theta)) * (K @ K)


def rotor_apply(R: Rotor3, v: np.ndarray) -> np.ndarray:
    """Apply a rotor to a 3-vector via conjugation v' = R v R̃.

    Equivalent to applying the corresponding rotation matrix.
    """
    v = np.asarray(v, dtype=float)
    if v.shape != (3,):
        raise ValueError(f"v must be a 3-vector, got shape {v.shape}")
    M = rotor_to_rotation_matrix(R)
    return M @ v


def rotor_compose(R1: Rotor3, R2: Rotor3) -> Rotor3:
    """Geometric product of two rotors. (R1 R2 corresponds to applying
    R2 first, then R1.)

    For rotors with s + b12 e_{12} + b13 e_{13} + b23 e_{23}, the
    product rules in Cl(3,0) give the formula below. Norm is preserved.
    """
    s1, (b12_1, b13_1, b23_1) = R1.s, R1.b
    s2, (b12_2, b13_2, b23_2) = R2.s, R2.b
    s_out = (s1 * s2
             - b12_1 * b12_2 - b13_1 * b13_2 - b23_1 * b23_2)
    b12 = (s1 * b12_2 + s2 * b12_1
           - b13_1 * b23_2 + b23_1 * b13_2)
    b13 = (s1 * b13_2 + s2 * b13_1
           + b12_1 * b23_2 - b23_1 * b12_2)
    b23 = (s1 * b23_2 + s2 * b23_1
           - b12_1 * b13_2 + b13_1 * b12_2)
    return Rotor3(float(s_out), np.array([b12, b13, b23]))


# ---------------------------------------------------------------------
# SVD → (rotor, spectrum) — Sudjianto's bridge
# ---------------------------------------------------------------------

def svd_to_rotor(M: np.ndarray) -> Tuple[Rotor3, np.ndarray, Rotor3]:
    """Convert SVD of M ∈ R^{3×3} to (rotor_left, spectrum, rotor_right).

    M = U Σ V^T factorises as M = R_L · diag(σ) · R_R^T with R_L, R_R
    orthogonal. We map R_L → rotor_L ∈ Spin(3), R_R → rotor_R ∈ Spin(3),
    and return the singular spectrum σ separately. Sign flips are
    absorbed to keep both rotors in SO(3).

    Returns (rotor_L, σ, rotor_R) such that
        M v = rotor_L · diag(σ) · (rotor_R^T · v)
    where the rotor actions are conjugations.
    """
    M = np.asarray(M, dtype=float)
    if M.shape != (3, 3):
        raise ValueError(f"M must be 3x3, got {M.shape}")
    U, sigma, Vt = np.linalg.svd(M, full_matrices=True)

    # Ensure both U, V have det = +1 (proper rotations) by absorbing a
    # sign into σ if necessary.
    detU = float(np.linalg.det(U))
    detV = float(np.linalg.det(Vt))
    if detU < 0:
        U[:, -1] = -U[:, -1]
        sigma[-1] = -sigma[-1]
    if detV < 0:
        Vt[-1, :] = -Vt[-1, :]
        sigma[-1] = -sigma[-1]

    rotor_L = rotation_matrix_to_rotor(U)
    rotor_R = rotation_matrix_to_rotor(Vt.T)
    return rotor_L, sigma, rotor_R


# ---------------------------------------------------------------------
# Rotor-based emergent rank (boundedness-preserving)
# ---------------------------------------------------------------------

def rotor_emergent_rank(M: np.ndarray, T: int,
                          threshold: float = 2.0
                          ) -> Tuple[int, np.ndarray, Rotor3, Rotor3]:
    """Emergent rank diagnostic in rotor representation.

    Parameters
    ----------
    M : (3, 3) error-correction matrix Π̂.
    T : sample size.
    threshold : κ for σ_i √T > κ.

    Returns
    -------
    rank : int, number of effective singular values.
    sigma : (3,) singular spectrum.
    rotor_L, rotor_R : rotors of left / right singular vectors. Each
        has unit norm by construction; this is the boundedness
        guarantee of Sudjianto.
    """
    rotor_L, sigma, rotor_R = svd_to_rotor(M)
    sqrtT = float(np.sqrt(T))
    rank = int(np.sum(sigma * sqrtT > threshold))
    return rank, sigma, rotor_L, rotor_R
