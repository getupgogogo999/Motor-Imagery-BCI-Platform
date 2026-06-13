"""
快速系统化实验（增量保存，可中断续跑）
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

from config import GDF_DIR, OUTPUTS_DIR, RANDOM_STATE
from run_experiments import (
    BAND_OPTIONS,
    METHODS,
    WINDOW_OPTIONS,
    run_leakage_audit,
    run_single_experiment,
    summarize_results,
)
from src.gdf_diagnostics import print_diagnostics_summary, run_diagnostics

RESULTS_PATH = OUTPUTS_DIR / "experiments" / "all_results.csv"


def load_done_keys(df: pd.DataFrame) -> set:
    if df.empty:
        return set()
    return set(
        zip(df["subject"], df["method"], df["band"], df["window"], df["car"])
    )


def run_fast(subjects: list, methods: list | None = None):
    out_dir = OUTPUTS_DIR / "experiments"
    out_dir.mkdir(parents=True, exist_ok=True)

    if RESULTS_PATH.exists():
        results = pd.read_csv(RESULTS_PATH)
    else:
        results = pd.DataFrame()

    done = load_done_keys(results)
    methods = methods or ["CSP+SVM", "FBCSP+LDA"]
    rows = results.to_dict("records") if not results.empty else []

    total = len(subjects) * len(BAND_OPTIONS) * len(WINDOW_OPTIONS) * 2 * len(methods)
    print(f"计划实验: {total}，已完成: {len(done)}")

    for subject in subjects:
        for band_name, bandpass in BAND_OPTIONS.items():
            for window_name, (tmin, tmax) in WINDOW_OPTIONS.items():
                for use_car in [False, True]:
                    for method in methods:
                        key = (subject, method, band_name, window_name, use_car)
                        if key in done:
                            continue
                        try:
                            save_cm = (
                                subject in ("A01", "A04", "A06")
                                and band_name == "8-30Hz"
                                and window_name == "0.5-4.0s"
                                and method == "FBCSP+LDA"
                            )
                            row = run_single_experiment(
                                subject, band_name, bandpass, window_name,
                                tmin, tmax, use_car, method, out_dir, save_cm=save_cm,
                            )
                            rows.append(row)
                            pd.DataFrame(rows).to_csv(RESULTS_PATH, index=False)
                            done.add(key)
                            print(
                                f"{subject} {method} {band_name} {window_name} CAR={use_car} "
                                f"-> acc={row['accuracy']:.3f} kappa={row['kappa']:.3f} "
                                f"[{len(done)}/{total}]"
                            )
                        except Exception as exc:
                            print(f"FAIL {key}: {exc}")

    df = pd.DataFrame(rows)
    if df.empty:
        print("无结果")
        return df

    df.to_csv(RESULTS_PATH, index=False)
    summarize_results(df)

    # 默认配置对比表
    default = df[
        (df["band"] == "8-30Hz") & (df["window"] == "0.5-2.5s") & (df["car"] == False)
    ].pivot(index="subject", columns="method", values="accuracy")
    default["mean"] = default.mean(axis=1)
    default.to_csv(out_dir / "default_config_comparison.csv")
    print(f"\n结果: {RESULTS_PATH}")
    return df


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--subjects", default="all")
    args = parser.parse_args()

    if args.subjects == "all":
        subjects = [f"A{i:02d}" for i in range(1, 10)]
    else:
        subjects = [f"A{int(s.strip()):02d}" for s in args.subjects.split(",")]

    print("=" * 60)
    print("  BCI 2a 快速系统化实验（增量保存）")
    print("=" * 60)

    run_leakage_audit("A01")
    diag = run_diagnostics(GDF_DIR)
    print_diagnostics_summary(diag)

    run_fast(subjects)


if __name__ == "__main__":
    main()
