"""
demo_pca_rpca.py
----------------
Demonstrates adaptive control concepts using PCA and Robust PCA (RPCA)
on the synthetic gas turbine dataset.

Key ideas
---------
PCA   : captures the low-dimensional manifold of normal operation.
        The first few principal components encode the dominant thermodynamic
        correlations (power setting, ambient temperature).

RPCA  : decomposes  X = L + S  where
        L (low-rank) → nominal thermodynamic correlations
        S (sparse)   → abrupt sensor faults / anomalies

Adaptive control loop (concept)
  1. MEASURE   sensor vector x_k
  2. DECOMPOSE X_window = L + S  via RPCA
  3. PROJECT   z_k = L_k · V_r  (low-rank PC scores = control state)
  4. FAULT     if ||S_k|| > θ  → isolate channel, raise alert
  5. REGIME    cluster z_k     → select gain schedule / physics model
  6. ADAPT     re-fit L when regime shift |Δz| > δ detected (degradation)
  7. CONTROL   emit u_k = K(regime) · z_k

Usage
-----
    python demo_pca_rpca.py

Requires
    numpy, pandas, matplotlib
    turbine_data.csv  (run generate_dataset.py first)
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


# ─────────────────────────────────────────────────────────────────────────────
# 1.  Robust PCA  –  Inexact Augmented Lagrangian Method
#     Reference: Lin, Chen, Ma (2010) "The Augmented Lagrange Multiplier Method
#                for Exact Recovery of Corrupted Low-Rank Matrices"
# ─────────────────────────────────────────────────────────────────────────────
def _svt(X: np.ndarray, tau: float) -> tuple[np.ndarray, int]:
    """Singular Value Thresholding operator  D_τ(X)."""
    U, s, Vt  = np.linalg.svd(X, full_matrices=False)
    s_thresh  = np.maximum(s - tau, 0.0)
    return U @ np.diag(s_thresh) @ Vt, int(np.sum(s_thresh > 0))


def _soft(X: np.ndarray, tau: float) -> np.ndarray:
    """Element-wise soft (shrinkage) thresholding operator  S_τ(X)."""
    return np.sign(X) * np.maximum(np.abs(X) - tau, 0.0)


def rpca(
    X: np.ndarray,
    lam: float | None = None,
    tol: float = 1e-7,
    max_iter: int = 1_000,
    verbose: bool = False,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Robust PCA  –  decomposes  X = L + S.

    Solves:   min  ||L||_*  +  λ ||S||_1
              s.t. L + S = X

    via the Inexact ALM algorithm (Lin et al. 2010).

    Parameters
    ----------
    X        : (m, n) data matrix
    lam      : sparse regularisation weight  [default 1 / √max(m,n)]
    tol      : convergence tolerance  ||X − L − S||_F / ||X||_F
    max_iter : maximum ALM iterations
    verbose  : print iteration log

    Returns
    -------
    L : (m, n) low-rank component
    S : (m, n) sparse component
    """
    m, n  = X.shape
    lam   = lam or 1.0 / np.sqrt(max(m, n))

    # ALM penalty parameters
    mu     = m * n / (4.0 * np.sum(np.abs(X)))
    mu_bar = mu * 1e7
    rho    = 1.5                              # penalty growth rate

    norm_X = np.linalg.norm(X, 'fro')
    Y      = X / max(np.linalg.norm(X, 2),   # dual variable initialisation
                     np.max(np.abs(X)) / lam)
    L = np.zeros_like(X)
    S = np.zeros_like(X)

    for k in range(max_iter):
        L, rank = _svt(X - S + Y / mu, 1.0 / mu)
        S        = _soft(X - L + Y / mu,  lam / mu)
        residual = X - L - S
        Y       += mu * residual
        mu       = min(rho * mu, mu_bar)

        err = np.linalg.norm(residual, 'fro') / norm_X
        if verbose and k % 100 == 0:
            print(f"    iter {k:4d}  err={err:.2e}  rank={rank}")
        if err < tol:
            if verbose:
                print(f"    Converged at iter {k}  rank={rank}")
            break

    return L, S


# ─────────────────────────────────────────────────────────────────────────────
# 2.  PCA helper
# ─────────────────────────────────────────────────────────────────────────────
def run_pca(
    X: np.ndarray,
    n_components: int = 3,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Classical PCA via full SVD.

    Returns
    -------
    scores     : (m, n_components) principal component scores
    components : (n_components, n) loading vectors
    residuals  : (m,) per-sample reconstruction residual  ||x - x̂||₂
    s          : (min(m,n),) all singular values (for explained variance)
    """
    U, s, Vt   = np.linalg.svd(X, full_matrices=False)
    components = Vt[:n_components]
    scores     = X @ components.T
    X_recon    = scores @ components
    residuals  = np.linalg.norm(X - X_recon, axis=1)
    return scores, components, residuals, s


# ─────────────────────────────────────────────────────────────────────────────
# 3.  Load and z-score normalise
# ─────────────────────────────────────────────────────────────────────────────
SENSOR_COLS = ['Nc', 'PR', 'T2', 'TIT', 'EGT', 'Wf',
               'eta_c', 'eta_t', 'W_net', 'SFC']


def load_and_scale(path: str = 'turbine_data.csv'):
    df      = pd.read_csv(path, index_col='sample_id')
    X_raw   = df[SENSOR_COLS].values.astype(float)
    mu      = X_raw.mean(axis=0)
    sigma   = X_raw.std(axis=0)
    X_norm  = (X_raw - mu) / sigma
    return df, X_norm, mu, sigma


# ─────────────────────────────────────────────────────────────────────────────
# 4.  Visualisation  (6-panel figure)
# ─────────────────────────────────────────────────────────────────────────────
COLOUR = {
    'normal':      '#2196F3',   # blue
    'degradation': '#FF9800',   # orange
    'transient':   '#4CAF50',   # green
    'fault':       '#F44336',   # red
}


def plot_results(
    df: pd.DataFrame,
    X_norm: np.ndarray,
    scores_pca: np.ndarray,
    res_pca: np.ndarray,
    L: np.ndarray,
    S: np.ndarray,
    scores_rpca: np.ndarray,
    res_rpca: np.ndarray,
) -> None:
    fault_mask = (df['fault_channel'] != 'none').values
    mode       = df['mode'].values

    fig = plt.figure(figsize=(16, 14))
    fig.suptitle(
        'Gas Turbine Adaptive Control Dataset  –  PCA vs RPCA',
        fontsize=14, fontweight='bold', y=0.99,
    )
    gs = fig.add_gridspec(3, 2, hspace=0.50, wspace=0.35)

    # ── (a) Cumulative explained variance ─────────────────────────────────
    ax = fig.add_subplot(gs[0, 0])
    _, s_X, _ = np.linalg.svd(X_norm, full_matrices=False)
    _, s_L, _ = np.linalg.svd(L,      full_matrices=False)
    k         = min(10, len(s_X))
    ev_X = np.cumsum(s_X**2) / np.sum(s_X**2) * 100
    ev_L = np.cumsum(s_L**2) / np.sum(s_L**2) * 100
    ax.plot(range(1, k + 1), ev_X[:k], 'o-', color='#607D8B', label='Raw X  (PCA)')
    ax.plot(range(1, k + 1), ev_L[:k], 's-', color='#9C27B0', label='L matrix (RPCA)')
    ax.axhline(95, ls='--', color='gray', lw=0.8, label='95 % threshold')
    ax.set(
        xlabel='Number of principal components',
        ylabel='Cumulative variance [%]',
        title='(a) Explained variance: raw X vs. RPCA low-rank L',
    )
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

    # ── (b) PC1 vs PC2 coloured by operating mode ─────────────────────────
    ax = fig.add_subplot(gs[0, 1])
    for m, col in [('normal',      COLOUR['normal']),
                   ('degradation', COLOUR['degradation']),
                   ('transient',   COLOUR['transient'])]:
        mask = mode == m
        ax.scatter(scores_rpca[mask, 0], scores_rpca[mask, 1],
                   s=4, alpha=0.5, color=col, label=m)
    ax.scatter(
        scores_rpca[fault_mask, 0], scores_rpca[fault_mask, 1],
        s=25, color=COLOUR['fault'], marker='x', label='fault', zorder=5,
    )
    ax.set(
        xlabel='PC1  (power setting / load)',
        ylabel='PC2  (ambient temperature)',
        title='(b) RPCA scores – operating regime separation',
    )
    ax.legend(fontsize=8, markerscale=2)
    ax.grid(alpha=0.3)

    # ── (c) Degradation drift along PC1 ───────────────────────────────────
    ax   = fig.add_subplot(gs[1, 0])
    dmsk = mode == 'degradation'
    ax.scatter(
        df['degradation'].values[dmsk],
        scores_rpca[dmsk, 0],
        s=5, alpha=0.7, color=COLOUR['degradation'],
    )
    # linear trend line
    x_deg = df['degradation'].values[dmsk]
    z     = np.polyfit(x_deg, scores_rpca[dmsk, 0], 1)
    p     = np.poly1d(z)
    ax.plot(np.sort(x_deg), p(np.sort(x_deg)),
            '--', color='#BF360C', lw=1.5, label=f'trend  slope={z[0]:.2f}')
    ax.set(
        xlabel='Degradation fraction  (0 = clean, 1 = fouled)',
        ylabel='PC1 score',
        title='(c) PC1 shift tracks compressor fouling (gradual degradation)',
    )
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

    # ── (d) Reconstruction residuals – PCA vs RPCA ────────────────────────
    ax  = fig.add_subplot(gs[1, 1])
    idx = np.arange(len(res_pca))
    ax.plot(idx, res_pca,  lw=0.5, alpha=0.7, color='#607D8B', label='PCA residual')
    ax.plot(idx, res_rpca, lw=0.5, alpha=0.7, color='#9C27B0', label='RPCA residual')
    ax.scatter(
        np.where(fault_mask)[0], res_pca[fault_mask],
        s=15, color=COLOUR['fault'], marker='x', zorder=5, label='known faults',
    )
    ax.set(
        xlabel='Sample index',
        ylabel='‖residual‖₂',
        title='(d) Reconstruction residuals: PCA bloated by faults, RPCA clean',
    )
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

    # ── (e) RPCA sparse component – fault isolation ────────────────────────
    ax = fig.add_subplot(gs[2, 0])
    S_mag = np.linalg.norm(S, axis=1)
    thr   = np.percentile(S_mag[~fault_mask], 97)
    ax.plot(S_mag, lw=0.5, color='#607D8B', alpha=0.7, label='‖S row‖₂')
    ax.axhline(thr, ls='--', color='gray', lw=1.0,
               label=f'97th-pct threshold = {thr:.3f}')
    ax.scatter(
        np.where(fault_mask)[0], S_mag[fault_mask],
        s=15, color=COLOUR['fault'], marker='x', zorder=5, label='known faults',
    )
    ax.set(
        xlabel='Sample index',
        ylabel='‖S row‖₂',
        title='(e) RPCA sparse component isolates abrupt sensor faults',
    )
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

    # ── (f) Adaptive control concept sketch ───────────────────────────────
    ax = fig.add_subplot(gs[2, 1])
    ax.axis('off')
    concept = (
        "  ADAPTIVE CONTROL LOOP  (concept)\n"
        "  ─────────────────────────────────────────────\n"
        "\n"
        "  1. MEASURE\n"
        "       sensor vector  x_k  (10 channels)\n"
        "\n"
        "  2. DECOMPOSE  (RPCA, sliding window)\n"
        "       X_window = L + S\n"
        "\n"
        "  3. PROJECT\n"
        "       z_k = L_k · V_r   (r PC scores)\n"
        "       → compact, fault-free state estimate\n"
        "\n"
        "  4. FAULT CHECK\n"
        "       if ‖S_k‖ > θ  → isolate channel\n"
        "                      → raise maintenance alert\n"
        "\n"
        "  5. REGIME ID\n"
        "       cluster z_k  → select gain schedule\n"
        "                       or physics model\n"
        "\n"
        "  6. ADAPT  (on degradation detection)\n"
        "       if ‖Δz‖ > δ  → re-fit PCA subspace\n"
        "                       → update control gains\n"
        "\n"
        "  7. CONTROL\n"
        "       emit  u_k = K(regime) · z_k"
    )
    ax.text(
        0.03, 0.97, concept,
        transform=ax.transAxes,
        va='top', ha='left', fontsize=9,
        fontfamily='monospace',
        bbox=dict(boxstyle='round', fc='#F5F5F5', ec='#BDBDBD', lw=1),
    )
    ax.set_title('(f) Adaptive control concept', fontsize=10)

    out = 'pca_rpca_demo.png'
    plt.savefig(out, dpi=150, bbox_inches='tight')
    print(f"\nPlot saved → {out}")
    plt.show()


# ─────────────────────────────────────────────────────────────────────────────
# 5.  Main
# ─────────────────────────────────────────────────────────────────────────────
def main() -> None:
    data_path = 'turbine_data.csv'
    if not Path(data_path).exists():
        sys.exit(
            f"ERROR: '{data_path}' not found.\n"
            "Run  python generate_dataset.py  first."
        )

    print("Loading dataset …")
    df, X_norm, mu, sigma = load_and_scale(data_path)
    m, n = X_norm.shape
    print(f"  {m:,} samples  ×  {n} sensor channels")

    # ── PCA on raw (fault-corrupted) data ─────────────────────────────────
    print("\nRunning PCA on raw X …")
    scores_pca, comps_pca, res_pca, s_pca = run_pca(X_norm, n_components=3)
    n_pca = np.searchsorted(
        np.cumsum(s_pca**2) / np.sum(s_pca**2), 0.95
    ) + 1
    print(f"  Components for 95 % variance (raw X): {n_pca}")

    # ── RPCA ──────────────────────────────────────────────────────────────
    print("\nRunning RPCA (Inexact ALM) … this may take ~15 s")
    L, S = rpca(X_norm, verbose=True)
    rank_L = int(np.linalg.matrix_rank(L, tol=1e-3))
    nnz_S  = int(np.sum(np.abs(S) > 1e-4))
    print(f"\n  rank(L) = {rank_L}")
    print(f"  nnz(S)  = {nnz_S:,}  ({nnz_S / (m * n) * 100:.2f} % of entries)")

    _, s_L, _ = np.linalg.svd(L, full_matrices=False)
    n_rpca = np.searchsorted(
        np.cumsum(s_L**2) / np.sum(s_L**2), 0.95
    ) + 1
    print(f"  Components for 95 % variance (L):    {n_rpca}")

    # ── PCA on the clean low-rank matrix L ────────────────────────────────
    scores_rpca, comps_rpca, res_rpca, _ = run_pca(L, n_components=3)

    # ── Fault detection summary ────────────────────────────────────────────
    fault_mask = (df['fault_channel'] != 'none').values
    n_faults   = fault_mask.sum()

    thr_pca  = np.percentile(res_pca[~fault_mask],  97)
    thr_rpca = np.percentile(res_rpca[~fault_mask], 97)
    S_mag    = np.linalg.norm(S, axis=1)
    thr_S    = np.percentile(S_mag[~fault_mask],    97)

    dr_pca  = res_pca[fault_mask].gt(thr_pca).mean()   if hasattr(res_pca[fault_mask],  'gt') else np.mean(res_pca[fault_mask]  > thr_pca)
    dr_rpca = np.mean(res_rpca[fault_mask] > thr_rpca)
    dr_S    = np.mean(S_mag[fault_mask]    > thr_S)

    print("\n── Fault detection (97th-percentile threshold on clean samples) ──")
    print(f"  Known fault samples            : {n_faults}")
    print(f"  PCA  reconstruction residual   : {dr_pca*100:5.1f} %")
    print(f"  RPCA reconstruction residual   : {dr_rpca*100:5.1f} %")
    print(f"  RPCA sparse ‖S‖  detection rate: {dr_S*100:5.1f} %")
    print(
        "\n  → RPCA sparse component cleanly separates faults from nominal\n"
        "    thermodynamic variation, enabling reliable fault isolation\n"
        "    without disrupting the control state estimate z_k = L · V_r."
    )

    plot_results(
        df, X_norm,
        scores_pca, res_pca,
        L, S,
        scores_rpca, res_rpca,
    )


if __name__ == '__main__':
    main()
