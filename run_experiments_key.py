"""精简实验：关键配置 x 9 受试者，快速产出汇总表"""
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

from config import OUTPUTS_DIR
from run_experiments import (
    BAND_OPTIONS,
    WINDOW_OPTIONS,
    run_single_experiment,
    summarize_results,
)

KEY_WINDOWS = ["0.5-2.5s", "0.5-4.0s", "1.0-4.0s"]
KEY_BANDS = list(BAND_OPTIONS.keys())
subjects = [f"A{i:02d}" for i in range(1, 10)]
methods = ["CSP+SVM", "FBCSP+LDA"]
out_dir = OUTPUTS_DIR / "experiments"

rows = []
for subject in subjects:
    for band_name in KEY_BANDS:
        bandpass = BAND_OPTIONS[band_name]
        for window_name in KEY_WINDOWS:
            tmin, tmax = WINDOW_OPTIONS[window_name]
            for use_car in [False, True]:
                for method in methods:
                    row = run_single_experiment(
                        subject, band_name, bandpass, window_name,
                        tmin, tmax, use_car, method, out_dir,
                        save_cm=(window_name == "0.5-4.0s" and band_name == "8-30Hz" and method == "FBCSP+LDA"),
                    )
                    rows.append(row)
                    print(f"{subject} {method} {band_name} {window_name} CAR={use_car} -> {row['accuracy']:.3f}")

df = pd.DataFrame(rows)
df.to_csv(out_dir / "all_results_key.csv", index=False)
summarize_results(df)
print("Saved:", out_dir / "all_results_key.csv")
