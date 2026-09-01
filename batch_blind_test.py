"""
Batch validation against 7 simulated Blind Evaluation Datasets
(BLIND_01-06 = leak scenarios, BLIND_07 = no-leak control).
Mirrors the official scoring approach: average detection + localization
accuracy over the 6 leak scenarios; BLIND_07 scored only on false-positive
robustness. Replace `make_dataset(...)` calls with real organizer files
(pd.read_csv(...)) once they're issued.
"""
import pandas as pd
from robustness_test import make_dataset
from analytics_engine import run_analysis, score_localization, gradient_confirmation

# Simulated blind scenarios - swap for real files on the day
BLIND_SCENARIOS = [
    ("BLIND_01", 800, 0.05),
    ("BLIND_02", 2600, 0.08),
    ("BLIND_03", 4300, 0.05),
    ("BLIND_04", 6100, 0.15),
    ("BLIND_05", 7900, 0.05),
    ("BLIND_06", 9500, 0.10),
    ("BLIND_07", None, 0.05),  # no-leak control
]

print(f"{'Dataset':10s} {'Detected':>9} {'GradOK':>7} {'X_calc(m)':>10} {'Error%':>8} {'FalsePos':>9}")
print("-" * 65)

errors = []
detections_ok = 0
false_positive = False

for name, x_leak, noise in BLIND_SCENARIOS:
    df = make_dataset(leak_x_m=x_leak, noise_std=noise)
    result = run_analysis(df)
    detected = result.t_in_s is not None and result.t_out_s is not None
    grad_ok = gradient_confirmation(df, "inlet_pressure_bar", result.t_in_s) if detected else False

    if x_leak is None:
        false_positive = detected
        print(f"{name:10s} {'--':>9} {'--':>7} {'--':>10} {'--':>8} {('YES' if false_positive else 'no'):>9}")
    else:
        detections_ok += int(detected)
        if detected:
            err = score_localization(result.leak_coordinate_m, x_leak)
            errors.append(err)
            print(f"{name:10s} {'yes':>9} {('yes' if grad_ok else 'no'):>7} {result.leak_coordinate_m:10.1f} {err:8.3f} {'no':>9}")
        else:
            print(f"{name:10s} {'MISSED':>9} {'--':>7} {'--':>10} {'--':>8} {'no':>9}")

print("-" * 65)
print(f"Detection rate (6 leak scenarios): {detections_ok}/6")
print(f"Mean localization error:           {sum(errors)/len(errors):.3f}%" if errors else "N/A")
print(f"False positive on no-leak control: {'YES - FIX REQUIRED' if false_positive else 'NO - robust'}")
