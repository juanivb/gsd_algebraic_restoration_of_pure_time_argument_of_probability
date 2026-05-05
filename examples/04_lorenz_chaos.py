"""
Example 04 — Multivariate quadratic on chaotic systems.

Reproduces Table 6 of the paper: the multivariate quadratic basis
fits Lorenz '63 and coupled-logistic chaos to floating-point precision
(in-sample RMSE ~ 1e-13 to 1e-15).
"""
import numpy as np


def quad_design_mv(Y):
    Y = np.asarray(Y, dtype=float)
    T, d = Y.shape
    if T < 4:
        return np.zeros((0, 1)), np.zeros((0, d))
    Y_t = Y[1:-1]
    dY_t = Y[1:-1] - Y[:-2]
    Y_next = Y[2:]
    feats = [np.ones((Y_t.shape[0], 1)), Y_t, dY_t,
             Y_t * Y_t, dY_t * dY_t]
    for j in range(d):
        for k in range(j + 1, d):
            feats.append((Y_t[:, j] * Y_t[:, k]).reshape(-1, 1))
    for j in range(d):
        for k in range(j + 1, d):
            feats.append((dY_t[:, j] * dY_t[:, k]).reshape(-1, 1))
    for j in range(d):
        for k in range(d):
            feats.append((Y_t[:, j] * dY_t[:, k]).reshape(-1, 1))
    return np.hstack(feats), Y_next


def lorenz(T=5000, dt=0.01, sigma=10, rho=28, beta=8/3):
    Y = np.empty((T, 3)); Y[0] = (1, 1, 1)
    for t in range(1, T):
        x, y, z = Y[t-1]
        Y[t, 0] = x + dt * sigma * (y - x)
        Y[t, 1] = y + dt * (x * (rho - z) - y)
        Y[t, 2] = z + dt * (x * y - beta * z)
    return Y


def coupled_logistic(T=5000, r=3.95, eps=0.05):
    Y = np.empty((T, 2)); Y[0] = (0.4, 0.6)
    for t in range(1, T):
        y1, y2 = Y[t-1]
        f1 = r * y1 * (1 - y1); f2 = r * y2 * (1 - y2)
        Y[t, 0] = (1 - eps) * f1 + eps * f2
        Y[t, 1] = (1 - eps) * f2 + eps * f1
    return Y


def evaluate(label, Y):
    X, Y_next = quad_design_mv(Y)
    coef, *_ = np.linalg.lstsq(X, Y_next, rcond=None)
    pred = X @ coef
    rmse = float(np.sqrt(np.mean(np.sum((pred - Y_next)**2, axis=1))))
    print(f"  {label:>30s}:  in-sample RMSE = {rmse:.4e}")


def main():
    print("=" * 70)
    print(" Multivariate quadratic chaos recovery")
    print("=" * 70)
    print("\nNoise-free in-sample RMSE per time step:")
    evaluate("Lorenz '63 (σ=10, ρ=28, β=8/3)", lorenz())
    evaluate("Coupled logistic (r=3.95, ε=0.05)", coupled_logistic())

    print("\nNoisy (σ=0.05) in-sample RMSE:")
    rng = np.random.default_rng(42)
    Y_noisy = lorenz() + 0.05 * rng.standard_normal((5000, 3))
    evaluate("Lorenz + noise σ=0.05", Y_noisy)
    print()


if __name__ == "__main__":
    main()
