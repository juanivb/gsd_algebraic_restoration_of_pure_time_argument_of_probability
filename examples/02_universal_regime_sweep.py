"""
Example 02 — Universal regime sweep.

Reproduces Table 3 of the paper: the same OLS regression in the
pure-time basis recovers the AR coefficient across stationary,
near-unit-root, integrated, explosive, and AR(2) regimes.
"""
import numpy as np
from gsd_puretime import ptls_universal, ab_to_phi, var2_ols


def gen_ar1(phi, T, sigma, rng):
    eps = sigma * rng.standard_normal(T)
    y = np.zeros(T)
    for k in range(1, T):
        y[k] = phi * y[k-1] + eps[k]
    return y


def gen_ar2(phi1, phi2, T, sigma, rng):
    eps = sigma * rng.standard_normal(T)
    y = np.zeros(T)
    for k in range(2, T):
        y[k] = phi1 * y[k-1] + phi2 * y[k-2] + eps[k]
    return y


def main():
    T = 1500
    rng = np.random.default_rng(42)
    print("=" * 90)
    print(" Universal regime sweep — pure-time forward equation")
    print("=" * 90)
    print(f"{'regime':>26s} | {'truth (φ_1, φ_2)':>20s} | "
          f"{'PT-univ (φ_1, φ_2)':>22s} | {'OLS-AR(2)':>20s}")
    print("-" * 96)

    cells = [
        ("AR(1) φ=0.5",          (0.5, 0.0),
         lambda: gen_ar1(0.5, T, 1.0, rng)),
        ("AR(1) φ=0.95",         (0.95, 0.0),
         lambda: gen_ar1(0.95, T, 1.0, rng)),
        ("Random walk",          (1.0, 0.0),
         lambda: np.cumsum(rng.standard_normal(T))),
        ("AR(1) φ=1.05 explosive", (1.05, 0.0),
         lambda: gen_ar1(1.05, T, 1.0, rng)),
        ("AR(2) (1.4, -0.6)",    (1.4, -0.6),
         lambda: gen_ar2(1.4, -0.6, T, 1.0, rng)),
        ("AR(2) (0.9, 0.05)",    (0.9, 0.05),
         lambda: gen_ar2(0.9, 0.05, T, 1.0, rng)),
        ("I(2)",                 (None, None),
         lambda: np.cumsum(np.cumsum(rng.standard_normal(T)))),
    ]
    for label, truth, gen in cells:
        y = gen()
        a, b = ptls_universal(y)
        phi1_pt, phi2_pt = ab_to_phi(a, b)
        Phi1_ar, Phi2_ar = var2_ols(y.reshape(-1, 1))
        truth_str = (f"({truth[0]:+.3f}, {truth[1]:+.3f})"
                     if truth[0] is not None else "—")
        print(f"{label:>26s} | {truth_str:>20s} | "
              f"({phi1_pt:+8.4f}, {phi2_pt:+8.4f}) | "
              f"({float(Phi1_ar[0,0]):+8.4f}, {float(Phi2_ar[0,0]):+8.4f})")
    print()


if __name__ == "__main__":
    main()
