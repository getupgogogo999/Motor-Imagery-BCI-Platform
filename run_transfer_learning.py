"""
Cross-Subject Transfer Learning 实验（A04 / A06）

运行:
    python run_transfer_learning.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

from config import OUTPUTS_DIR
from src.transfer_learning import TRANSFER_CONFIGS, run_transfer_for_target

OUT_DIR = OUTPUTS_DIR / "experiments" / "transfer_learning"
RESULTS_CSV = OUT_DIR / "transfer_learning_results.csv"
BEST_CSV = OUT_DIR / "transfer_learning_best.csv"


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    all_rows = []

    for target_key in ["A04", "A06"]:
        all_rows.extend(run_transfer_for_target(target_key))

    df = pd.DataFrame(all_rows)
    df.to_csv(RESULTS_CSV, index=False)

    # 每个 target 最佳实验
    best_rows = []
    print(f"\n{'=' * 55}")
    print("  最佳迁移结果")
    print(f"{'=' * 55}")
    for target in ["A04", "A06"]:
        sub = df[(df["target"] == target) & df["accuracy"].notna()]
        if sub.empty:
            continue
        best = sub.loc[sub["accuracy"].idxmax()]
        best_rows.append(best)
        baseline = TRANSFER_CONFIGS[target]["baseline_cv"]
        print(f"  {target}: {best['experiment']} → {best['accuracy']:.1%} "
              f"(基线 {baseline:.1%}, Δ={best['improvement_pp']:+.1f}pp)")

    pd.DataFrame(best_rows).to_csv(BEST_CSV, index=False)
    print(f"\n结果: {RESULTS_CSV}")
    print(f"最佳: {BEST_CSV}")


if __name__ == "__main__":
    main()
