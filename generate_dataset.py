"""
generate_dataset.py
-------------------
Synthetic gas turbine dataset for demonstrating adaptive control
using PCA and Robust PCA (RPCA).

Operating phenomena covered
    1. Normal operating envelope  – 5 power settings × 3 ambient temperatures
    2. Gradual degradation        – compressor fouling + turbine erosion ramp
    3. Abrupt faults              – sparse sensor spikes (RPCA sparse component)
    4. Transient manoeuvres       – first-order lag dynamics between power steps

Data matrix structure (suitable for RPCA decomposition)
    X = L + S + N
    L : low-rank  – normal correlated sensor behaviour
    S : sparse    – abrupt faults on individual channels
    N : small i.i.d. Gaussian noise

Sensor channels (10 total)
    Nc     – corrected shaft speed          [%]
    PR     – compressor pressure ratio      [-]
    T2     – compressor delivery temp       [K]
    TIT    – turbine inlet temperature      [K]
    EGT    – exhaust gas temperature        [K]
    Wf     – fuel flow                      [g/s  (normalised)]
    eta_c  – compressor isentropic eff.     [-]
    eta_t  – turbine isentropic eff.        [-]
    W_net  – specific net work              [kJ/kg]
    SFC    – specific fuel consumption      [mg/kJ]

Usage
    python generate_dataset.py
    → writes turbine_data.csv in the current directory
"""

import numpy as np
import pandas as pd

# ─────────────────────────────────────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────────────────────────────────────
GAMMA = 1.4        # ratio of specific heats (air)
CP    = 1.005      # specific heat at constant pressure  [kJ/(kg·K)]
LHV   = 43_000    # lower heating value of kerosene      [kJ/kg]
ETA_B = 0.995      # combustion efficiency (fixed)
RNG   = np.random.default_rng(42)


# ─────────────────────────────────────────────────────────────────────────────
# 1. Simplified single-spool thermodynamic cycle
# ─────────────────────────────────────────────────────────────────────────────
def cycle(P_set: float, T_amb: float, eta_c: float, eta_t: float) -> dict:
    """
    Simplified single-spool gas turbine thermodynamic cycle.

    Parameters
    ----------
    P_set  : power setting             [0.60 – 1.00]
    T_amb  : ambient temperature       [K]
    eta_c  : compressor isentropic efficiency
    eta_t  : turbine    isentropic efficiency

    Returns
    -------
    dict mapping sensor name → value
    """
    exp = (GAMMA - 1.0) / GAMMA   # γ-1 / γ  ≈ 0.2857

    # --- Compressor --------------------------------------------------------
    PR   = 5.0 + 25.0 * P_set                       # pressure ratio  8 – 30
    T2   = T_amb * (1.0 + (PR**exp - 1.0) / eta_c)  # compressor delivery temp
    Nc   = 60.0 + 40.0 * P_set                      # corrected shaft speed [%]

    # --- Combustor ---------------------------------------------------------
    TIT  = 1_000.0 + 600.0 * P_set                  # turbine inlet temp [K]

    # --- Turbine -----------------------------------------------------------
    # Single-spool: turbine expands across the full pressure ratio
    T4   = TIT * (1.0 - eta_t * (1.0 - PR**(-exp))) # turbine exit temp
    EGT  = T4                                        # EGT ≈ turbine exit temp

    # --- Fuel flow (per unit air mass flow) --------------------------------
    Wf   = CP * (TIT - T2) / (ETA_B * LHV)          # [kg_fuel / kg_air]

    # --- Specific work and SFC ---------------------------------------------
    W_turbine    = CP * eta_t * (TIT - T4)
    W_compressor = CP * (T2 - T_amb)
    W_net        = W_turbine - W_compressor          # [kJ/kg_air]
    SFC          = Wf / max(W_net, 1e-6) * 1e6      # [mg/kJ]  (fuel per net work)

    return dict(
        Nc=Nc,
        PR=PR,
        T2=T2,
        TIT=TIT,
        EGT=EGT,
        Wf=Wf * 1e3,        # converted to g/s (×1000 for readability)
        eta_c=eta_c,
        eta_t=eta_t,
        W_net=W_net,
        SFC=SFC,
    )


# ─────────────────────────────────────────────────────────────────────────────
# 2. Normal operating envelope
# ─────────────────────────────────────────────────────────────────────────────
def generate_normal(n_per_condition: int = 120) -> list[dict]:
    """
    5 power settings × 3 ambient temperatures = 15 operating conditions.
    Each condition has n_per_condition samples with small Gaussian noise.
    """
    power_settings = [0.60, 0.70, 0.80, 0.90, 1.00]
    ambient_temps  = [273.15, 288.15, 303.15]   # ISA−15 / ISA / ISA+15  [K]
    eta_c0, eta_t0 = 0.880, 0.900               # clean-engine efficiencies

    rows = []
    for P in power_settings:
        for T in ambient_temps:
            for _ in range(n_per_condition):
                ec  = eta_c0 + RNG.normal(0, 0.001)
                et  = eta_t0 + RNG.normal(0, 0.001)
                row = cycle(P, T, ec, et)
                row.update(
                    P_setting=P, T_amb=T,
                    mode='normal', degradation=0.0,
                    fault_channel='none', fault_magnitude=0.0,
                )
                rows.append(row)
    return rows


# ─────────────────────────────────────────────────────────────────────────────
# 3. Gradual degradation
# ─────────────────────────────────────────────────────────────────────────────
def generate_degradation(n_steps: int = 500) -> list[dict]:
    """
    Linear fouling / erosion ramp (ISO 11086 style):
      eta_c drops  5 %  (compressor fouling)
      eta_t drops  2 %  (turbine blade erosion)
    over n_steps samples at a fixed cruise power setting.
    """
    P      = 0.85     # fixed cruise-like power setting
    T      = 288.15   # ISA day
    eta_c0 = 0.880
    eta_t0 = 0.900

    rows = []
    for i in range(n_steps):
        frac  = i / (n_steps - 1)                          # 0 (clean) → 1 (fouled)
        eta_c = eta_c0 - 0.050 * frac + RNG.normal(0, 0.001)
        eta_t = eta_t0 - 0.020 * frac + RNG.normal(0, 0.001)
        row   = cycle(P, T, eta_c, eta_t)
        row.update(
            P_setting=P, T_amb=T,
            mode='degradation', degradation=frac,
            fault_channel='none', fault_magnitude=0.0,
        )
        rows.append(row)
    return rows


# ─────────────────────────────────────────────────────────────────────────────
# 4. Transient manoeuvres
# ─────────────────────────────────────────────────────────────────────────────
def generate_transients(n_per_step: int = 80, tau: float = 0.15) -> list[dict]:
    """
    Ramp sequences between power steps using a first-order lag response.

    tau controls how quickly the engine responds (as a fraction of step length).
    Smaller tau → faster response.
    """
    steps  = [0.60, 1.00, 0.70, 0.90, 0.65]   # acceleration / deceleration profile
    T      = 288.15
    eta_c0 = 0.880
    eta_t0 = 0.900

    rows = []
    for idx in range(len(steps) - 1):
        P_start = steps[idx]
        P_end   = steps[idx + 1]
        for k in range(n_per_step):
            t   = k / n_per_step
            # First-order lag: P_actual approaches P_end with time constant tau
            P   = P_end - (P_end - P_start) * np.exp(-t / tau)
            ec  = eta_c0 + RNG.normal(0, 0.001)
            et  = eta_t0 + RNG.normal(0, 0.001)
            row = cycle(P, T, ec, et)
            row.update(
                P_setting=P, T_amb=T,
                mode='transient', degradation=0.0,
                fault_channel='none', fault_magnitude=0.0,
            )
            rows.append(row)
    return rows


# ─────────────────────────────────────────────────────────────────────────────
# 5. Abrupt fault injection  (sparse corruptions for RPCA)
# ─────────────────────────────────────────────────────────────────────────────
FAULT_CHANNELS = ['Nc', 'PR', 'T2', 'TIT', 'EGT', 'Wf', 'W_net', 'SFC']
FAULT_PROB     = 0.012   # ~1.2 % of samples carry a spike


def inject_faults(rows: list[dict]) -> list[dict]:
    """
    In-place: randomly spike a single sensor channel per affected sample.
    Spike magnitude = 3–8× channel standard deviation (clearly anomalous).
    """
    channel_scales = {
        ch: float(np.std([r[ch] for r in rows]))
        for ch in FAULT_CHANNELS
    }
    for row in rows:
        if RNG.random() < FAULT_PROB:
            ch   = RNG.choice(FAULT_CHANNELS)
            sign = RNG.choice([-1, 1])
            mag  = sign * RNG.uniform(3.0, 8.0) * channel_scales[ch]
            row[ch]               += mag
            row['fault_channel']   = ch
            row['fault_magnitude'] = mag
    return rows


# ─────────────────────────────────────────────────────────────────────────────
# 6. Assemble and save
# ─────────────────────────────────────────────────────────────────────────────
SENSOR_COLS = ['Nc', 'PR', 'T2', 'TIT', 'EGT', 'Wf',
               'eta_c', 'eta_t', 'W_net', 'SFC']
META_COLS   = ['P_setting', 'T_amb', 'mode', 'degradation',
               'fault_channel', 'fault_magnitude']


def build_dataset(out_path: str = 'turbine_data.csv') -> pd.DataFrame:
    """Build the full dataset and write it to CSV."""
    rows  = []
    rows += generate_normal(n_per_condition=120)
    rows += generate_degradation(n_steps=500)
    rows += generate_transients(n_per_step=80)

    inject_faults(rows)   # sparse spikes injected across all modes

    df = pd.DataFrame(rows, columns=SENSOR_COLS + META_COLS)
    df.index.name = 'sample_id'
    df.to_csv(out_path)

    n_faults = (df['fault_channel'] != 'none').sum()
    print(f"Dataset saved → {out_path}")
    print(f"  Total samples : {len(df):,}")
    print(f"  Fault samples : {n_faults} ({n_faults/len(df)*100:.1f} %)")
    print()
    print(df.groupby('mode').size().rename('count').to_string())
    return df


if __name__ == '__main__':
    build_dataset()
