"""
Example 01 — Battery D: deterministic exactness.

Reproduces the central claim of §4 of the paper: on noise-free
deterministic DGPs, the pure-time estimator recovers parameters to
numerical precision (modulo the optimiser).

Three DGPs:
  * Sine: y_t = sin(0.3 t + 0.5)            → AR(2) form (φ_1=2cos(0.3), φ_2=-1)
  * Logistic chaos r=3.95                   → quadratic form recovers r
  * Hénon (a=1.4, b=0.3)                    → quadratic form recovers (a, b)

Run:
    python examples/01_battery_d_deterministic.py
"""
import numpy as np

from gsd_puretime import (
    ptls_universal, ab_to_phi,
    ptls_quadratic, henon_params_from_quadratic, logistic_param_from_quadratic,
)


def main():
    print("=" * 70)
    print(" Battery D — Deterministic exactness")
    print("=" * 70)

    # --- Sine ---
    omega = 0.3
    T = 500
    t = np.arange(T)
    y = np.sin(omega * t + 0.5)
    a, b = ptls_universal(y)
    phi1, phi2 = ab_to_phi(a, b)
    print(f"\nSine y_t = sin({omega} t + 0.5):")
    print(f"  Recovered:  φ_1 = {phi1:.10f},  φ_2 = {phi2:.10f}")
    print(f"  Truth:      φ_1 = {2*np.cos(omega):.10f},  φ_2 = -1.0000000000")
    print(f"  |Δφ_1| = {abs(phi1 - 2*np.cos(omega)):.2e},  "
          f"|Δφ_2| = {abs(phi2 - (-1)):.2e}")

    # --- Logistic chaos ---
    r_true = 3.95
    y = np.empty(T); y[0] = 0.4
    for k in range(1, T):
        y[k] = r_true * y[k-1] * (1 - y[k-1])
    # 1D-driven chaos: the full 6-feature quadratic basis is rank-deficient
    # (the trajectory lives on a 1D manifold). Use the natural reduced basis
    # (1, y_t, y_t^2) directly via OLS:
    y_t = y[:-1]
    y_next = y[1:]
    X = np.column_stack([np.ones_like(y_t), y_t, y_t * y_t])
    coef, *_ = np.linalg.lstsq(X, y_next, rcond=None)
    r_hat = 0.5 * (coef[1] - coef[2])           # average c_1 and -c_3 for stability
    print(f"\nLogistic y_{{t+1}} = {r_true} y_t (1 - y_t)  [reduced basis (1, y, y²)]:")
    print(f"  Recovered r = {r_hat:.10f}")
    print(f"  |Δr|        = {abs(r_hat - r_true):.2e}")

    # --- Hénon ---
    a_true, b_true = 1.4, 0.3
    y = np.empty(T); y[0] = 0; y[1] = 0
    for k in range(2, T):
        y[k] = 1.0 - a_true * y[k-1]**2 + b_true * y[k-2]
    c = ptls_quadratic(y)
    a_hat, b_hat = henon_params_from_quadratic(c)
    print(f"\nHénon y_{{t+1}} = 1 - {a_true} y_t² + {b_true} y_{{t-1}}:")
    print(f"  Recovered:  a = {a_hat:.12f},  b = {b_hat:.12f}")
    print(f"  |Δa| = {abs(a_hat - a_true):.2e},  |Δb| = {abs(b_hat - b_true):.2e}")
    print()


if __name__ == "__main__":
    main()
