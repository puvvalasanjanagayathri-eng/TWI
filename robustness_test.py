"""
Robustness test suite - simulates what the Blind Evaluation Datasets will
throw at the algorithm: noise, a no-leak control, and different leak
locations. This is what you show the jury as evidence of generalization
(the "Robustness" 15% + validates the "no false alarms" requirement).
"""
import numpy as np
import pandas as pd
from analytics_engine import run_analysis, score_localization, PIPELINE_LENGTH_M, WAVE_SPEED_MPS

rng = np.random.default_rng(42)


def make_dataset(leak_x_m=None, duration_s=12, noise_std=0.05, baseline_in=60.0, baseline_out=55.0):
    """
    leak_x_m = None  -> no-leak control scenario (BLIND_07-style)
    leak_x_m = value -> leak at that coordinate; back-calculates t_in/t_out
                        from the SAME NPW physics used for scoring, so this
                        is an honest simulation, not a rigged one.
    """
    n = int(duration_s * 1000 / 100) + 1
    t_ms = np.arange(0, n * 100, 100)
    inlet = baseline_in + rng.normal(0, noise_std, size=n)
    outlet = baseline_out + rng.normal(0, noise_std, size=n)

    if leak_x_m is not None:
        # physics: wave reaches inlet at t_in, outlet at t_out
        t_in = leak_x_m / WAVE_SPEED_MPS + 2.0          # +2s so there's baseline data first
        t_out = (PIPELINE_LENGTH_M - leak_x_m) / WAVE_SPEED_MPS + 2.0
        for i, tms in enumerate(t_ms):
            ts = tms / 1000.0
            if ts >= t_in:
                decay = min(1.0, (ts - t_in) * 3)  # ramps down over ~0.3s
                inlet[i] = baseline_in * (1 - 0.55 * decay) + rng.normal(0, noise_std)
            if ts >= t_out:
                decay = min(1.0, (ts - t_out) * 3)
                outlet[i] = baseline_out * (1 - 0.55 * decay) + rng.normal(0, noise_std)

    return pd.DataFrame({
        "relative_time_ms": t_ms,
        "inlet_pressure_bar": inlet,
        "outlet_pressure_bar": outlet,
    })


scenarios = [
    ("BLIND_A - leak near inlet (X=1000m)", 1000, 0.05),
    ("BLIND_B - leak mid-pipe (X=5000m)", 5000, 0.05),
    ("BLIND_C - leak near outlet (X=9000m)", 9000, 0.05),
    ("BLIND_D - noisy leak (X=3500m, high noise)", 3500, 0.25),
    ("BLIND_07-style - NO LEAK control", None, 0.05),
]

print(f"{'Scenario':45s} {'X_calc(m)':>10} {'Error%':>8} {'FalseAlarm':>11}")
print("-" * 80)
for name, x_leak, noise in scenarios:
    df = make_dataset(leak_x_m=x_leak, noise_std=noise)
    result = run_analysis(df)

    if x_leak is None:
        false_alarm = "YES" if result.t_in_s is not None and result.t_out_s is not None else "no"
        print(f"{name:45s} {'--':>10} {'--':>8} {false_alarm:>11}")
    else:
        if result.leak_coordinate_m is not None:
            err = score_localization(result.leak_coordinate_m, x_leak)
            print(f"{name:45s} {result.leak_coordinate_m:10.1f} {err:8.3f} {'no':>11}")
        else:
            print(f"{name:45s} {'MISSED':>10} {'--':>8} {'no':>11}")
