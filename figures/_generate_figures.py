"""
Generate the four canonical figures shown in the README.
Run from repo root:  python figures/_generate_figures.py
"""
from __future__ import annotations
import sys
from pathlib import Path

_HERE = Path(__file__).resolve().parent
_ROOT = _HERE.parent
sys.path.insert(0, str(_ROOT))

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

from gsd_puretime import (
    ptls_universal, ptls_quadratic, ptmv,
    ec_matrix, emergent_rank,
    ab_to_phi, henon_params_from_quadratic, var2_ols,
)

OUT = _HERE
plt.rcParams.update({
    "axes.grid": True, "grid.alpha": 0.25,
    "axes.spines.top": False, "axes.spines.right": False,
    "font.size": 10,
})


# ------------------------------------------------------------------
# Figure 1 — Forecast comparison across regimes (5 panels)
# ------------------------------------------------------------------
def gen_ar1(phi, T, sigma, rng):
    eps = sigma * rng.standard_normal(T); y = np.zeros(T)
    for k in range(1, T): y[k] = phi*y[k-1] + eps[k]
    return y
def gen_ar2(phi1, phi2, T, sigma, rng):
    eps = sigma * rng.standard_normal(T); y = np.zeros(T)
    for k in range(2, T): y[k] = phi1*y[k-1] + phi2*y[k-2] + eps[k]
    return y
def gen_rw(T, sigma, rng): return np.cumsum(sigma*rng.standard_normal(T))

T = 600
regimes = [
    ("AR(1) φ=0.5",       lambda r: gen_ar1(0.5, T, 1.0, r)),
    ("AR(1) φ=0.95",      lambda r: gen_ar1(0.95, T, 1.0, r)),
    ("Random walk",       lambda r: gen_rw(T, 1.0, r)),
    ("AR(1) φ=1.05 explosive", lambda r: gen_ar1(1.05, T, 1.0, r)),
    ("AR(2) (1.4, -0.6)", lambda r: gen_ar2(1.4, -0.6, T, 1.0, r)),
]

def rolling(y, method, train_frac=0.8):
    n = int(train_frac*len(y)); h = list(y[:n]); preds = []
    for y_true in y[n:]:
        ha = np.asarray(h)
        if method == "pt":
            a, b = ptls_universal(ha)
            pred = a*ha[-1] + b*(ha[-1]-ha[-2])
        else:
            P1, P2 = var2_ols(ha.reshape(-1, 1))
            pred = float(P1[0,0])*ha[-1] + float(P2[0,0])*ha[-2]
        preds.append(pred); h.append(y_true)
    return np.asarray(preds), y[n:]

fig, axes = plt.subplots(len(regimes), 1, figsize=(11, 12), sharex=False)
for ax, (label, gen) in zip(axes, regimes):
    rng = np.random.default_rng(7)
    y = gen(rng)
    p_pt, y_test = rolling(y, "pt")
    p_ar, _      = rolling(y, "ar")
    n_train = len(y) - len(y_test)
    idx = np.arange(n_train, len(y))
    rmse_pt = float(np.sqrt(np.mean((p_pt - y_test)**2)))
    rmse_ar = float(np.sqrt(np.mean((p_ar - y_test)**2)))
    ax.plot(idx, y_test, color="black", lw=1.7, label="truth", alpha=0.85)
    ax.plot(idx, p_pt, color="tab:blue", lw=1.2, label=f"PT-univ  (RMSE={rmse_pt:.3f})")
    ax.plot(idx, p_ar, color="tab:orange", lw=1.2, ls="--",
            label=f"OLS-AR(2)  (RMSE={rmse_ar:.3f})")
    ax.set_title(f"Regime: {label}", fontsize=11, loc="left")
    ax.legend(loc="best", fontsize=9, framealpha=0.9)
    ax.set_xlabel("t"); ax.set_ylabel("y")
fig.suptitle("One-step forecasts across regimes — synthetic data, train 80%/test 20%",
             fontsize=12, y=1.0)
plt.tight_layout()
plt.savefig(OUT / "forecast_regimes.png", dpi=140, bbox_inches="tight")
plt.close(fig)
print("✓ forecast_regimes.png")


# ------------------------------------------------------------------
# Figure 2 — Cointegration diagnostic (scatter + SVD bar)
# ------------------------------------------------------------------
T = 2000
rng = np.random.default_rng(11)
eps2 = 0.5*rng.standard_normal(T); y2 = np.cumsum(eps2)
u = np.zeros(T); eps_u = 0.2*rng.standard_normal(T)
for k in range(1, T): u[k] = 0.5*u[k-1] + eps_u[k]
y1 = 0.6*y2 + u
Y_coint = np.column_stack([y1, y2])
Y_indep = np.cumsum(rng.standard_normal((T, 2)), axis=0)

A_c, B_c = ptmv(Y_coint); M_c = ec_matrix(A_c, B_c)
A_i, B_i = ptmv(Y_indep); M_i = ec_matrix(A_i, B_i)
sig_c = np.linalg.svd(M_c, compute_uv=False)
sig_i = np.linalg.svd(M_i, compute_uv=False)

fig, axes = plt.subplots(1, 2, figsize=(13, 5))
ax = axes[0]
ax.scatter(Y_coint[:, 1], Y_coint[:, 0], s=4, alpha=0.45, color="tab:blue",
           label="Cointegrated  (band: y₁ ≈ 0.6 y₂)")
ax.scatter(Y_indep[:, 1], Y_indep[:, 0], s=4, alpha=0.35, color="tab:gray",
           label="Independent RWs (no shared structure)")
ax.set_xlabel("y₂"); ax.set_ylabel("y₁")
ax.set_title("Cointegration as a linear band in phase space")
ax.legend(loc="best", fontsize=10)

ax = axes[1]
labels = ["σ₁\ncointeg", "σ₂\ncointeg", "σ₁\nindep", "σ₂\nindep"]
vals = [sig_c[0]*np.sqrt(T), sig_c[1]*np.sqrt(T),
        sig_i[0]*np.sqrt(T), sig_i[1]*np.sqrt(T)]
colors = ["tab:blue", "tab:blue", "tab:gray", "tab:gray"]
bars = ax.bar(labels, vals, color=colors, alpha=0.85)
ax.axhline(2.5, color="red", ls="--", lw=1.4, label="threshold κ = 2.5")
for b, v in zip(bars, vals):
    ax.text(b.get_x()+b.get_width()/2, v + max(vals)*0.025,
            f"{v:.2f}", ha="center", fontsize=10)
ax.set_ylabel("σᵢ √T")
ax.set_title("Effective rank from σᵢ √T (Johansen-style)")
ax.legend(loc="best", fontsize=10)
fig.suptitle("Emergent cointegration rank from SVD of Π̂ = Â + B̂ − I",
             fontsize=12, y=1.02)
plt.tight_layout()
plt.savefig(OUT / "cointegration_svd_gap.png", dpi=140, bbox_inches="tight")
plt.close(fig)
print("✓ cointegration_svd_gap.png")


# ------------------------------------------------------------------
# Figure 3 — Hénon attractor: observed vs recovered
# ------------------------------------------------------------------
T = 1500
y = np.zeros(T); y[0] = 0; y[1] = 0
for k in range(2, T): y[k] = 1.0 - 1.4*y[k-1]**2 + 0.3*y[k-2]
c = ptls_quadratic(y); a_h, b_h = henon_params_from_quadratic(c)
y_rec = np.zeros(T); y_rec[0] = 0; y_rec[1] = 0
for k in range(2, T): y_rec[k] = 1.0 - a_h*y_rec[k-1]**2 + b_h*y_rec[k-2]

fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))
axes[0].scatter(y[:-1], y[1:], s=3, alpha=0.6, color="tab:blue")
axes[0].set_xlabel("y_{t-1}"); axes[0].set_ylabel("y_t")
axes[0].set_title("Observed Hénon attractor  (a=1.4, b=0.3)")
axes[1].scatter(y_rec[:-1], y_rec[1:], s=3, alpha=0.6, color="tab:green")
axes[1].set_xlabel("y_{t-1}"); axes[1].set_ylabel("y_t")
axes[1].set_title(f"Recovered: a={a_h:.10f}, b={b_h:.10f}")
fig.suptitle("Hénon parameters recovered to ~10⁻¹⁵ from a single OLS regression",
             fontsize=12, y=1.0)
plt.tight_layout()
plt.savefig(OUT / "henon_recovery.png", dpi=140, bbox_inches="tight")
plt.close(fig)
print("✓ henon_recovery.png")


# ------------------------------------------------------------------
# Figure 4 — Lorenz butterfly: observed vs recovered
# ------------------------------------------------------------------
def lorenz(T, dt=0.01, sigma=10.0, rho=28.0, beta=8/3, init=(1, 1, 1)):
    Y = np.zeros((T, 3)); Y[0] = init
    for t in range(1, T):
        x, y, z = Y[t-1]
        Y[t, 0] = x + dt*sigma*(y-x)
        Y[t, 1] = y + dt*(x*(rho-z)-y)
        Y[t, 2] = z + dt*(x*y-beta*z)
    return Y

def quad_design_mv(Y):
    Y_t = Y[1:-1]; dY_t = Y[1:-1]-Y[:-2]; Y_next = Y[2:]
    d = Y.shape[1]
    feats = [np.ones((Y_t.shape[0], 1)), Y_t, dY_t, Y_t*Y_t, dY_t*dY_t]
    for j in range(d):
        for k in range(j+1, d):
            feats.append((Y_t[:, j]*Y_t[:, k]).reshape(-1, 1))
    for j in range(d):
        for k in range(j+1, d):
            feats.append((dY_t[:, j]*dY_t[:, k]).reshape(-1, 1))
    for j in range(d):
        for k in range(d):
            feats.append((Y_t[:, j]*dY_t[:, k]).reshape(-1, 1))
    return np.hstack(feats), Y_next

T_lz = 5000
Y_obs = lorenz(T_lz)
X, Y_next = quad_design_mv(Y_obs)
coef, *_ = np.linalg.lstsq(X, Y_next, rcond=None)
pred = X @ coef
rmse = float(np.sqrt(np.mean(np.sum((pred - Y_next)**2, axis=1))))

def step_qm(Y_full, coef):
    y_t = Y_full[-1]; dy_t = Y_full[-1] - Y_full[-2]
    feat = [1.0, *y_t, *dy_t, *(y_t*y_t), *(dy_t*dy_t)]
    d = len(y_t)
    for j in range(d):
        for k in range(j+1, d): feat.append(y_t[j]*y_t[k])
    for j in range(d):
        for k in range(j+1, d): feat.append(dy_t[j]*dy_t[k])
    for j in range(d):
        for k in range(d): feat.append(y_t[j]*dy_t[k])
    return coef.T @ np.asarray(feat)

Y_rec = np.zeros_like(Y_obs); Y_rec[:2] = Y_obs[:2]
for t in range(2, T_lz):
    Y_rec[t] = step_qm(Y_rec[:t], coef)

fig = plt.figure(figsize=(13, 5.5))
ax1 = fig.add_subplot(1, 2, 1, projection="3d")
ax1.plot(Y_obs[:, 0], Y_obs[:, 1], Y_obs[:, 2], lw=0.4, color="tab:blue")
ax1.set_title(f"Observed Lorenz '63 (σ=10, ρ=28, β=8/3, T={T_lz})")
ax1.set_xlabel("x"); ax1.set_ylabel("y"); ax1.set_zlabel("z")

ax2 = fig.add_subplot(1, 2, 2, projection="3d")
ax2.plot(Y_rec[:, 0], Y_rec[:, 1], Y_rec[:, 2], lw=0.4, color="tab:green")
ax2.set_title(f"Regenerated from quadratic OLS  (in-sample RMSE = {rmse:.1e})")
ax2.set_xlabel("x"); ax2.set_ylabel("y"); ax2.set_zlabel("z")
fig.suptitle("Lorenz butterfly recovered to floating-point precision",
             fontsize=12, y=1.0)
plt.tight_layout()
plt.savefig(OUT / "lorenz_recovery.png", dpi=140, bbox_inches="tight")
plt.close(fig)
print("✓ lorenz_recovery.png")

print(f"\nAll four figures written to {OUT}/")
