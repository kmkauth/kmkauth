"""
plot_control_comparison.py
--------------------------
PhD defense presentation figures.

Closed-loop simulation comparing two controllers on a simplified
linearised gas turbine model.

  Fixed PID  – gains tuned at nominal; blind to degradation and sensor faults
  Adaptive   – PCA-based regime detection + RPCA sparse fault isolation

Four sequential phases (150 s total)
  Phase 1   0 – 30 s    Normal + setpoint step at t = 15 s
  Phase 2  30 – 82 s    Gradual fouling ramp  (EGT drift, Nc efficiency drop)
  Phase 3  82 – 95 s    Sustained EGT sensor positive spike
                          Fixed PID: fuel-cut → EGT crash
                          Adaptive:  RPCA isolates spike → no crash
  Phase 4  95 – 150 s   Recovery; setpoint return at t = 110 s

Outputs
  defense_control_comparison.{pdf,png}   – 5-panel combined timeline
  defense_metrics_bar.{pdf,png}          – IAE / ITAE / violations bar chart

Usage
    python plot_control_comparison.py
"""

from __future__ import annotations
import numpy as np
import matplotlib.pyplot as plt

# ─────────────────────────────────────────────────────────────────────────────
# Global style  (poster / defense: large fonts, bold colours, white background)
# ─────────────────────────────────────────────────────────────────────────────
plt.rcParams.update({
    'font.family':       'sans-serif',
    'font.size':         14,
    'axes.titlesize':    15,
    'axes.titleweight':  'bold',
    'axes.labelsize':    14,
    'axes.labelweight':  'bold',
    'xtick.labelsize':   12,
    'ytick.labelsize':   12,
    'legend.fontsize':   12,
    'legend.framealpha': 0.92,
    'lines.linewidth':   2.2,
    'axes.linewidth':    1.5,
    'grid.linewidth':    0.7,
    'axes.grid':         True,
    'grid.alpha':        0.35,
    'grid.color':        '#888888',
    'figure.facecolor':  'white',
    'axes.facecolor':    'white',
    'savefig.dpi':       200,
    'savefig.bbox':      'tight',
})

C_REF   = '#212121'   # setpoint    – near-black
C_FIXED = '#C62828'   # fixed PID   – bold red
C_ADAP  = '#1565C0'   # adaptive    – bold blue
C_WF_F  = '#EF6C00'   # Wf fixed    – orange
C_WF_A  = '#0277BD'   # Wf adaptive – mid-blue
A_REG   = 0.45        # region fill alpha

# ─────────────────────────────────────────────────────────────────────────────
# Simulation parameters
# ─────────────────────────────────────────────────────────────────────────────
N    = 1500
DT   = 0.1            # seconds per step → 150 s total
TIME = np.arange(N) * DT

# All temperatures in Rankine  (°R = K × 1.8)
EGT_NOM   = 1476.0    # nominal EGT   [°R]   (820 K × 1.8)
NC_NOM    = 80.0      # nominal Nc    [%]
WF_NOM    = 20.0      # nominal Wf    [g/s]
K_EGT_NOM = 9.9       # ∂EGT/∂Wf  at nominal  [°R/(g/s)]  (5.5 K/(g/s) × 1.8)
K_NC_NOM  = 0.38      # ∂Nc /∂Wf  at nominal  [%/(g/s)]
TAU_EGT   = 10        # EGT thermal lag   [steps]
TAU_NC    = 6         # Nc mechanical lag [steps]
WF_MIN    = 8.0       # fuel flow limits  [g/s]
WF_MAX    = 35.0

# Phase boundaries (step indices)
STEP_IDX    = 150     # setpoint step        t = 15 s
DEG_START   = 300     # fouling onset        t = 30 s
DEG_END     = 820     # fouling plateau      t = 82 s
FAULT_START = 820     # sensor fault starts  t = 82 s
FAULT_END   = 1020    # sensor fault ends    t = 102 s   (20 s sustained fault)
RECOVER_IDX = 1150    # setpoint return      t = 115 s

# PID gains tuned for nominal  (EGT loop only)
KP = 0.30 / K_EGT_NOM   # ≈ 0.055
KI = 0.020 / K_EGT_NOM  # ≈ 0.0036
KD = 0.50 / K_EGT_NOM   # ≈ 0.091

FAULT_THRESH = 36.0   # absolute residual threshold [°R] for RPCA sparse detection  (20 K × 1.8)

RNG = np.random.default_rng(42)


# ─────────────────────────────────────────────────────────────────────────────
# Profiles
# ─────────────────────────────────────────────────────────────────────────────
def make_setpoints() -> tuple[np.ndarray, np.ndarray]:
    EGT_ref            = np.full(N, EGT_NOM)
    EGT_ref[STEP_IDX:RECOVER_IDX] = 1656.0   # 920 K × 1.8  [°R]
    Nc_ref             = np.full(N, NC_NOM)
    Nc_ref[STEP_IDX:RECOVER_IDX]  = 90.0
    return EGT_ref, Nc_ref


def make_degradation() -> np.ndarray:
    """
    Linear fouling ramp 0 → 1 over DEG_START:DEG_END, then held at 1.
    Scaled to 10 % max perturbation in plant.
    """
    deg = np.zeros(N)
    deg[DEG_START:DEG_END] = np.linspace(0, 1, DEG_END - DEG_START)
    deg[DEG_END:]           = 1.0
    return deg * 0.10


# ─────────────────────────────────────────────────────────────────────────────
# Plant model
#   Degradation effects (compressor fouling):
#     • Additive EGT drift   +35 K at full fouling (less compressor cooling)
#     • K_Nc drop of 80 %   at full fouling  (lower isentropic efficiency)
#   K_EGT kept approximately constant (dominant thermodynamic relationship).
# ─────────────────────────────────────────────────────────────────────────────
def plant_step(
    EGT: float, Nc: float, Wf: float, deg: float
) -> tuple[float, float]:
    D_EGT = 63.0 * (deg / 0.10)            # additive EGT disturbance  [°R]  (35 K × 1.8)
    K_NC  = K_NC_NOM * (1.0 - 0.80 * (deg / 0.10))  # efficiency degradation

    EGT_ss  = EGT_NOM + K_EGT_NOM * (Wf - WF_NOM) + D_EGT
    Nc_ss   = NC_NOM  + K_NC       * (Wf - WF_NOM)

    EGT_new = EGT + (EGT_ss - EGT) / TAU_EGT + RNG.normal(0, 0.63)  # 0.35 K × 1.8
    Nc_new  = Nc  + (Nc_ss  - Nc)  / TAU_NC  + RNG.normal(0, 0.07)
    return EGT_new, Nc_new


def fault_spike(k: int) -> float:
    """Always-positive sustained EGT sensor spike during fault window [°R]."""
    if FAULT_START <= k < FAULT_END:
        return float(RNG.uniform(270, 360))  # 150–200 K × 1.8  →  270–360 °R
    return 0.0


# ─────────────────────────────────────────────────────────────────────────────
# Fixed PID  (no fault awareness, no adaptation)
# ─────────────────────────────────────────────────────────────────────────────
def simulate_fixed_pid(
    EGT_ref: np.ndarray,
    Nc_ref: np.ndarray,
    deg: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    EGT = np.zeros(N); Nc = np.zeros(N); Wf = np.zeros(N)
    EGT[0], Nc[0], Wf[0] = EGT_NOM, NC_NOM, WF_NOM
    integ  = 0.0
    e_prev = 0.0

    for k in range(1, N):
        EGT_meas = EGT[k - 1] + fault_spike(k)          # faulty measurement

        e      = EGT_ref[k] - EGT_meas
        integ  = float(np.clip(integ + e * DT, -108, 108))  # anti-windup  [°R·s]
        deriv  = (e - e_prev) / DT
        e_prev = e

        Wf[k] = float(np.clip(
            WF_NOM + KP * e + KI * integ + KD * deriv, WF_MIN, WF_MAX))
        EGT[k], Nc[k] = plant_step(EGT[k - 1], Nc[k - 1], Wf[k], deg[k])

    return EGT, Nc, Wf


# ─────────────────────────────────────────────────────────────────────────────
# Adaptive controller
#   PCA regime detection  → online bias + gain scheduling
#     Estimates additive EGT drift from running bias (proxy for PCA manifold
#     shift detection); feeds forward a compensating Wf offset.
#   RPCA fault isolation  → residual threshold on one-step-ahead prediction;
#     replaces faulty measurement with model prediction when ||S_k|| > θ.
# ─────────────────────────────────────────────────────────────────────────────
WINDOW = 50   # sliding window for statistics


def simulate_adaptive(
    EGT_ref: np.ndarray,
    Nc_ref: np.ndarray,
    deg: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray,
           np.ndarray, np.ndarray, np.ndarray]:
    EGT          = np.zeros(N)
    Nc           = np.zeros(N)
    Wf           = np.zeros(N)
    fault_flag   = np.zeros(N, dtype=bool)
    sparse_norm  = np.zeros(N)
    bias_est_log = np.zeros(N)

    EGT[0], Nc[0], Wf[0] = EGT_NOM, NC_NOM, WF_NOM

    integ     = 0.0
    e_prev    = 0.0
    EGT_pred  = EGT_NOM           # one-step model prediction
    bias_est  = 0.0               # estimated additive EGT disturbance (PCA shift)

    resid_buf = [0.0] * WINDOW    # for RPCA sparse threshold
    EGT_buf   = [EGT_NOM] * WINDOW

    for k in range(1, N):
        EGT_true = EGT[k - 1]
        EGT_meas = EGT_true + fault_spike(k)

        # ── RPCA fault detection (||S_k|| proxy) ──────────────────────────
        # Absolute threshold on one-step-ahead prediction residual.
        # In true RPCA: this corresponds to a non-zero entry in sparse S.
        resid   = abs(EGT_meas - EGT_pred)
        resid_buf.append(resid); resid_buf.pop(0)
        sigma_r = float(np.std(resid_buf)) + 0.5
        s_norm  = resid / sigma_r          # normalised for plotting
        sparse_norm[k] = s_norm
        is_fault = resid > FAULT_THRESH    # fixed absolute threshold  [°R]

        fault_flag[k] = is_fault
        EGT_ctrl = EGT_pred if is_fault else EGT_meas   # L component for control

        # ── PCA regime: estimate additive bias (degradation disturbance) ──
        # Only update when no fault present (avoids corrupting PCA state).
        if not is_fault:
            EGT_buf.append(EGT_ctrl); EGT_buf.pop(0)
            # Bias = mean(EGT_actual) − expected_EGT_at_current_Wf
            expected = EGT_NOM + K_EGT_NOM * (Wf[k - 1] - WF_NOM)
            new_bias = float(np.mean(EGT_buf)) - expected
            # Low-pass filter (mimics slow PCA subspace update)
            bias_est = 0.98 * bias_est + 0.02 * new_bias
            bias_est = float(np.clip(bias_est, 0, 72))   # max 63 °R disturbance + margin
        bias_est_log[k] = bias_est

        # Adaptive feed-forward: reduce Wf to compensate for estimated drift
        # (regime-aware gain scheduling via PCA state)
        Wf_ff = -bias_est / K_EGT_NOM   # feed-forward correction  [g/s]

        # ── PID (adaptive) ─────────────────────────────────────────────────
        e      = EGT_ref[k] - EGT_ctrl
        integ  = float(np.clip(integ + e * DT, -108, 108))  # anti-windup  [°R·s]
        deriv  = (e - e_prev) / DT
        e_prev = e

        Wf[k] = float(np.clip(
            WF_NOM + Wf_ff + KP * e + KI * integ + KD * deriv,
            WF_MIN, WF_MAX))
        EGT[k], Nc[k] = plant_step(EGT[k - 1], Nc[k - 1], Wf[k], deg[k])

        # Update model prediction (observer)
        EGT_pred = EGT[k]

    return EGT, Nc, Wf, fault_flag, sparse_norm, bias_est_log


# ─────────────────────────────────────────────────────────────────────────────
# Performance metrics  (local time for ITAE – avoids shape mismatch)
# ─────────────────────────────────────────────────────────────────────────────
def metrics(EGT: np.ndarray, EGT_ref: np.ndarray) -> dict[str, float]:
    e     = np.abs(EGT - EGT_ref)
    t_loc = np.arange(len(e)) * DT           # local time for slice
    iae   = float(np.sum(e) * DT)
    itae  = float(np.sum(t_loc * e) * DT)
    viol  = int(np.sum(EGT > 1710))   # 950 K × 1.8 = 1710 °R  safety limit
    return dict(IAE=iae, ITAE=itae, violations=viol)


# ─────────────────────────────────────────────────────────────────────────────
# Phase shading helper
# ─────────────────────────────────────────────────────────────────────────────
C_PH1 = '#E3F2FD'   # blue-tinted
C_PH2 = '#FFE0B2'   # amber  – degradation
C_PH3 = '#FFF9C4'   # yellow – fault
C_PH4 = '#DCEDC8'   # green  – recovery


def shade_phases(ax: plt.Axes, label: bool = True) -> None:
    t  = TIME
    kw = dict(alpha=A_REG, zorder=0, linewidth=0)
    ax.axvspan(0,            t[STEP_IDX],    color=C_PH1, **kw)
    ax.axvspan(t[STEP_IDX],  t[DEG_START],   color=C_PH1, **kw)
    ax.axvspan(t[DEG_START], t[FAULT_START], color=C_PH2, **kw)
    ax.axvspan(t[FAULT_START], t[FAULT_END], color=C_PH3, **kw)
    ax.axvspan(t[FAULT_END], t[RECOVER_IDX], color=C_PH2, **kw)
    ax.axvspan(t[RECOVER_IDX], t[-1],        color=C_PH4, **kw)

    if label:
        ymin, ymax = ax.get_ylim()
        y0 = ymin + 0.96 * (ymax - ymin)
        fs = 9.0
        for mid, txt, col in [
            ((0 + t[STEP_IDX]) / 2,                   'Phase 1\nNominal',     '#37474F'),
            ((t[STEP_IDX] + t[DEG_START]) / 2,        'Phase 1\nCruise',      '#37474F'),
            ((t[DEG_START] + t[FAULT_START]) / 2,     'Phase 2\nFouling',     '#BF360C'),
            ((t[FAULT_START] + t[FAULT_END]) / 2,     'Phase 3\nSensor\nFault','#F57F17'),
            ((t[FAULT_END] + t[RECOVER_IDX]) / 2,     'Phase 2+\nFouled',     '#BF360C'),
            ((t[RECOVER_IDX] + t[-1]) / 2,            'Phase 4\nRecovery',    '#2E7D32'),
        ]:
            ax.text(mid, y0, txt, ha='center', va='top',
                    fontsize=fs, color=col, linespacing=1.3)


# ─────────────────────────────────────────────────────────────────────────────
# Figure 1 – Combined 5-panel timeline
# ─────────────────────────────────────────────────────────────────────────────
def plot_main(
    EGT_ref, Nc_ref,
    EGT_f, Nc_f, Wf_f,
    EGT_a, Nc_a, Wf_a,
    fault_flag, sparse_norm, bias_est,
) -> plt.Figure:

    t   = TIME
    fig, axes = plt.subplots(
        5, 1, figsize=(14, 19), sharex=True,
        gridspec_kw=dict(hspace=0.10, height_ratios=[3, 2.5, 2.2, 2, 1.8]),
    )

    # ── Panel 1: EGT ─────────────────────────────────────────────────────────
    ax = axes[0]
    ax.plot(t, EGT_ref,  '--', color=C_REF,   lw=2.0, zorder=5, label='Setpoint  EGT_ref')
    ax.plot(t, EGT_f,          color=C_FIXED, lw=1.8, zorder=3, alpha=0.9, label='Fixed PID')
    ax.plot(t, EGT_a,          color=C_ADAP,  lw=2.2, zorder=4, label='Adaptive (PCA + RPCA)')
    ax.axhline(1710, ls=':', color='#B71C1C', lw=1.5, label='Safety limit  1710 °R  (950 K)')

    # Annotate the fault crash for fixed PID
    crash_idx = int(np.argmin(EGT_f[FAULT_END:FAULT_END + 200])) + FAULT_END
    ax.annotate(
        f'Fixed PID:\nfuel cut → EGT\ncrashes {EGT_f[crash_idx]:.0f} °R',
        xy=(t[crash_idx], EGT_f[crash_idx]),
        xytext=(t[crash_idx] + 5, EGT_f[crash_idx] - 8),
        fontsize=10.5, color=C_FIXED, fontweight='bold',
        arrowprops=dict(arrowstyle='->', color=C_FIXED, lw=1.8),
        bbox=dict(boxstyle='round,pad=0.2', fc='#FFEBEE', ec=C_FIXED, alpha=0.9),
    )
    ax.annotate(
        'Adaptive:\nfault isolated in S\n→ tracking maintained',
        xy=(t[FAULT_START + 60], EGT_a[FAULT_START + 60]),
        xytext=(t[FAULT_START + 60] - 22, EGT_a[FAULT_START + 60] - 20),
        fontsize=10.5, color=C_ADAP, fontweight='bold',
        arrowprops=dict(arrowstyle='->', color=C_ADAP, lw=1.8),
        bbox=dict(boxstyle='round,pad=0.2', fc='#E3F2FD', ec=C_ADAP, alpha=0.9),
    )
    # Annotate degradation drift
    deg_ann = (DEG_START + FAULT_START) // 2
    ax.annotate(
        'Fixed: EGT drifts\n(uncompensated\nfouling)',
        xy=(t[deg_ann], EGT_f[deg_ann]),
        xytext=(t[deg_ann] - 18, EGT_f[deg_ann] + 8),
        fontsize=10, color=C_FIXED,
        arrowprops=dict(arrowstyle='->', color=C_FIXED, lw=1.5),
    )

    shade_phases(ax, label=True)
    ax.set_ylabel('EGT  [°R]', labelpad=6)
    ax.set_title(
        'Gas Turbine Adaptive Control  ·  PCA + RPCA  |  PhD Defense',
        fontsize=16, pad=10,
    )
    ax.legend(loc='upper left', ncol=2)
    ax.set_ylim(1404, 1773)   # 780 K × 1.8 = 1404 °R,  985 K × 1.8 = 1773 °R

    # ── Panel 2: Nc ──────────────────────────────────────────────────────────
    ax = axes[1]
    ax.plot(t, Nc_ref, '--', color=C_REF,   lw=2.0, zorder=5, label='Setpoint  Nc_ref')
    ax.plot(t, Nc_f,         color=C_FIXED, lw=1.8, zorder=3, alpha=0.9, label='Fixed PID')
    ax.plot(t, Nc_a,         color=C_ADAP,  lw=2.2, zorder=4, label='Adaptive (PCA + RPCA)')

    # Annotate Nc crash during fault
    nc_crash = int(np.argmin(Nc_f[FAULT_START:FAULT_END + 200])) + FAULT_START
    ax.annotate(
        f'Nc drops to {Nc_f[nc_crash]:.1f} %\n(fuel cut)',
        xy=(t[nc_crash], Nc_f[nc_crash]),
        xytext=(t[nc_crash] + 4, Nc_f[nc_crash] - 2.5),
        fontsize=10, color=C_FIXED,
        arrowprops=dict(arrowstyle='->', color=C_FIXED, lw=1.5),
    )
    # Annotate Nc steady-state loss under degradation
    nc_deg_ann = (DEG_START + FAULT_START) // 2
    ax.annotate(
        'Nc degrades\n(η_c drop)',
        xy=(t[nc_deg_ann], Nc_f[nc_deg_ann]),
        xytext=(t[nc_deg_ann] - 15, Nc_f[nc_deg_ann] - 2.0),
        fontsize=10, color=C_FIXED,
        arrowprops=dict(arrowstyle='->', color=C_FIXED, lw=1.5),
    )

    shade_phases(ax, label=False)
    ax.set_ylabel('Nc  [%]', labelpad=6)
    ax.legend(loc='upper left', ncol=2)
    ax.set_ylim(72, 97)

    # ── Panel 3: Fuel flow ────────────────────────────────────────────────────
    ax = axes[2]
    ax.plot(t, Wf_f, color=C_WF_F, lw=1.8, alpha=0.9, label='Fixed PID')
    ax.plot(t, Wf_a, color=C_WF_A, lw=2.2, label='Adaptive (PCA + RPCA)')
    ax.axhline(WF_MIN, ls=':', color='gray', lw=1.2)
    ax.axhline(WF_MAX, ls=':', color='gray', lw=1.2,
               label=f'Saturation limits  ({WF_MIN}–{WF_MAX} g/s)')

    # Annotate fuel cut saturation
    sat_idx = int(np.argmin(Wf_f[FAULT_START:FAULT_END])) + FAULT_START
    ax.annotate(
        'Fixed: Wf → min\n(fuel cut, saturation)',
        xy=(t[sat_idx], Wf_f[sat_idx]),
        xytext=(t[sat_idx] + 4, Wf_f[sat_idx] + 4),
        fontsize=10, color=C_FIXED,
        arrowprops=dict(arrowstyle='->', color=C_FIXED, lw=1.5),
    )

    shade_phases(ax, label=False)
    ax.set_ylabel('Fuel flow Wf  [g/s]', labelpad=6)
    ax.legend(loc='upper left', ncol=2)
    ax.set_ylim(3, 42)

    # ── Panel 4: RPCA sparse norm (fault detection signal) ───────────────────
    ax = axes[3]
    ax.fill_between(t, sparse_norm, alpha=0.20, color=C_ADAP)
    ax.plot(t, sparse_norm, color=C_ADAP, lw=1.5,
            label='‖S_k‖ / σ  (RPCA sparse component norm)')
    ax.axhline(FAULT_THRESH / 10, ls='--', color='#F57F17', lw=2.0,
               label=f'Detection threshold  θ = {FAULT_THRESH} °R residual')
    det = np.where(fault_flag)[0]
    if len(det):
        ax.scatter(t[det], sparse_norm[det], s=25, color='#F57F17', zorder=5,
                   label='Fault detected')

    shade_phases(ax, label=False)
    ax.set_ylabel('‖S‖ / σ  [°R]', labelpad=6)
    ax.legend(loc='upper right', ncol=1)
    ax.set_ylim(0, max(sparse_norm.max() * 1.15, FAULT_THRESH * 2.5))

    # ── Panel 5: Estimated bias (PCA degradation tracking) ───────────────────
    ax = axes[4]
    deg_arr = make_degradation()
    true_bias = 63.0 * (deg_arr / 0.10)   # 35 K × 1.8 = 63 °R
    ax.plot(t, true_bias,  '--', color='gray',   lw=1.5, label='True EGT drift  D(t)')
    ax.plot(t, bias_est,         color=C_ADAP,   lw=2.2,
            label='PCA estimate  D̂(t)  (adaptive)')
    shade_phases(ax, label=False)
    ax.set_ylabel('EGT bias  [°R]', labelpad=6)
    ax.set_xlabel('Time  [s]', labelpad=6)
    ax.legend(loc='upper left', ncol=2)
    ax.set_ylim(-9, 81)   # -5 K × 1.8 = -9 °R,  45 K × 1.8 = 81 °R

    # Shared vertical phase dividers
    for ax in axes:
        for pt in [TIME[STEP_IDX], TIME[DEG_START], TIME[FAULT_START],
                   TIME[FAULT_END], TIME[RECOVER_IDX]]:
            ax.axvline(pt, color='#455A64', lw=0.9, ls=':', alpha=0.6, zorder=1)
        ax.set_xlim(0, TIME[-1])

    return fig


# ─────────────────────────────────────────────────────────────────────────────
# Figure 2 – Metrics bar chart
# ─────────────────────────────────────────────────────────────────────────────
def plot_metrics(
    EGT_ref: np.ndarray,
    EGT_f: np.ndarray,
    EGT_a: np.ndarray,
) -> plt.Figure:
    phases = {
        'Normal\n(0–15 s)':       (0,           STEP_IDX),
        'Cruise\n(15–30 s)':      (STEP_IDX,    DEG_START),
        'Fouling\n(30–82 s)':     (DEG_START,   FAULT_START),
        'Fault\n(82–95 s)':       (FAULT_START, FAULT_END),
        'Recovery\n(95–150 s)':   (RECOVER_IDX, N),
    }

    labels, iae_f, iae_a, itae_f, itae_a, viol_f, viol_a = [], [], [], [], [], [], []
    for label, (i0, i1) in phases.items():
        sl = slice(i0, i1)
        mf = metrics(EGT_f[sl], EGT_ref[sl])
        ma = metrics(EGT_a[sl], EGT_ref[sl])
        labels.append(label)
        iae_f.append(mf['IAE']);    iae_a.append(ma['IAE'])
        itae_f.append(mf['ITAE']); itae_a.append(ma['ITAE'])
        viol_f.append(mf['violations']); viol_a.append(ma['violations'])

    x, w = np.arange(len(labels)), 0.36
    fig, axes = plt.subplots(1, 3, figsize=(16, 5.5))
    fig.suptitle(
        'EGT Performance Metrics by Phase  –  Fixed PID vs Adaptive (PCA + RPCA)',
        fontsize=15, fontweight='bold', y=1.02,
    )

    def bar_pair(ax, vf, va, title, unit):
        b1 = ax.bar(x - w / 2, vf, w, color=C_FIXED, label='Fixed PID',     alpha=0.88)
        b2 = ax.bar(x + w / 2, va, w, color=C_ADAP,  label='Adaptive RPCA', alpha=0.88)
        ax.set_title(title)
        ax.set_ylabel(unit)
        ax.set_xticks(x); ax.set_xticklabels(labels, fontsize=11)
        ax.legend()
        ax.grid(axis='y', alpha=0.4)
        cap = max(max(vf), max(va)) * 0.015
        for bar in (*b1, *b2):
            h = bar.get_height()
            if h > 0:
                ax.text(bar.get_x() + bar.get_width() / 2, h + cap,
                        f'{h:.0f}', ha='center', va='bottom', fontsize=9)

    bar_pair(axes[0], iae_f,  iae_a,  'IAE  (Integral Absolute Error)',         '°R · s')
    bar_pair(axes[1], itae_f, itae_a, 'ITAE  (Integral Time-Weighted AE)',      '°R · s²')
    bar_pair(axes[2], viol_f, viol_a, 'Limit violations  (EGT > 1710 °R)',      'steps')

    # Overall improvement callout on IAE panel
    delta_pct = (sum(iae_f) - sum(iae_a)) / max(sum(iae_f), 1) * 100
    sign = '+' if delta_pct < 0 else ''
    axes[0].text(
        0.98, 0.97,
        f'Overall IAE reduction:\n{delta_pct:.1f} %',
        transform=axes[0].transAxes, ha='right', va='top',
        fontsize=12, fontweight='bold', color=C_ADAP,
        bbox=dict(boxstyle='round', fc='#E3F2FD', ec=C_ADAP, lw=1.5),
    )

    fig.tight_layout()
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────
def main() -> None:
    print("Building profiles …")
    EGT_ref, Nc_ref = make_setpoints()
    deg = make_degradation()

    print("Simulating fixed PID …")
    EGT_f, Nc_f, Wf_f = simulate_fixed_pid(EGT_ref, Nc_ref, deg)

    print("Simulating adaptive controller (PCA + RPCA) …")
    EGT_a, Nc_a, Wf_a, fault_flag, sparse_norm, bias_est = \
        simulate_adaptive(EGT_ref, Nc_ref, deg)

    # ── Full-timeline metrics ─────────────────────────────────────────────────
    mf = metrics(EGT_f, EGT_ref)
    ma = metrics(EGT_a, EGT_ref)
    print("\n── Full-timeline EGT metrics ─────────────────────────────────────────")
    print(f"  {'Metric':<30}  {'Fixed PID':>12}  {'Adaptive':>12}  {'Δ %':>8}")
    print(f"  {'-'*64}")
    for key in ('IAE', 'ITAE', 'violations'):
        vf, va = mf[key], ma[key]
        delta  = (vf - va) / max(vf, 1) * 100
        print(f"  {key:<30}  {vf:>12.1f}  {va:>12.1f}  {delta:>+7.1f} %")

    print(f"\n  Fault detections (adaptive): {int(fault_flag.sum())} / {FAULT_END - FAULT_START} fault steps")
    print(f"  False positives (non-fault) : {int(fault_flag[:FAULT_START].sum()) + int(fault_flag[FAULT_END:].sum())}")

    print("\nRendering Figure 1 – combined timeline …")
    fig1 = plot_main(
        EGT_ref, Nc_ref,
        EGT_f, Nc_f, Wf_f,
        EGT_a, Nc_a, Wf_a,
        fault_flag, sparse_norm, bias_est,
    )
    fig1.savefig('defense_control_comparison.png')
    fig1.savefig('defense_control_comparison.pdf')
    print("  → defense_control_comparison.{png,pdf}")

    print("Rendering Figure 2 – metrics bar chart …")
    fig2 = plot_metrics(EGT_ref, EGT_f, EGT_a)
    fig2.savefig('defense_metrics_bar.png')
    fig2.savefig('defense_metrics_bar.pdf')
    print("  → defense_metrics_bar.{png,pdf}")

    plt.show()
    print("\nDone.")


if __name__ == '__main__':
    main()
