"""
Example 03 — Emergent cointegration rank.

Reproduces Table 5 of the paper: the cointegration rank of a
multivariate panel is recovered from the SVD of $\\hat\\Pi = \\hat A
+ \\hat B - I$ with no prior specification.
"""
import numpy as np
from gsd_puretime import ptmv, ec_matrix, emergent_rank


def main():
    T = 1500
    rng = np.random.default_rng(11)
    print("=" * 80)
    print(" Emergent cointegration rank from SVD of Π̂")
    print("=" * 80)
    print(f"{'DGP':>32s} | {'σ_1 √T':>10s} | {'σ_2 √T':>10s} | "
          f"{'r̂':>3s} | truth")
    print("-" * 78)

    cells = []

    # M1: independent AR(1) — full rank
    eps = rng.standard_normal((T, 2))
    Y = np.zeros((T, 2))
    for k in range(1, T):
        Y[k, 0] = 0.5 * Y[k-1, 0] + eps[k, 0]
        Y[k, 1] = 0.7 * Y[k-1, 1] + eps[k, 1]
    cells.append(("M1: independent AR(1)", Y, 2))

    # M2: cointegrated I(1) — rank 1
    eps2 = 0.5 * rng.standard_normal(T); y2 = np.cumsum(eps2)
    u = np.zeros(T); eps_u = 0.2 * rng.standard_normal(T)
    for k in range(1, T):
        u[k] = 0.5 * u[k-1] + eps_u[k]
    y1 = 0.6 * y2 + u
    cells.append(("M2: cointegrated I(1)", np.column_stack([y1, y2]), 1))

    # M3: two independent random walks — rank 0
    cells.append(("M3: two independent RWs",
                  np.cumsum(rng.standard_normal((T, 2)), axis=0), 0))

    # M4: full stationary VAR(1) — full rank
    Phi1 = np.array([[0.5, 0.2], [0.15, 0.6]])
    Y = np.zeros((T, 2))
    for k in range(1, T):
        Y[k] = Phi1 @ Y[k-1] + rng.standard_normal(2)
    cells.append(("M4: full VAR(1) stationary", Y, 2))

    for label, Y, truth in cells:
        A, B = ptmv(Y)
        M = ec_matrix(A, B)
        r, sigmas = emergent_rank(M, T=T, threshold=2.5)
        sR = sigmas * np.sqrt(T)
        s1 = sR[0] if len(sR) > 0 else 0
        s2 = sR[1] if len(sR) > 1 else 0
        print(f"{label:>32s} | {s1:>10.3f} | {s2:>10.3f} | "
              f"{r:>3d} | {truth}")
    print()


if __name__ == "__main__":
    main()
