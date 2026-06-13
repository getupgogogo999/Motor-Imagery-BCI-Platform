"""
构建并验证「通用智能路由模型」motor_imagery_universal.pkl

用法:
    python run_build_universal_model.py
"""
from __future__ import annotations

import shutil
from pathlib import Path

import pandas as pd

from config import BEST_MODEL_PATH, GDF_DIR, MODELS_DIR, OUTPUTS_DIR
from src.universal_model import (
    ALL_SUBJECTS,
    build_universal_bundle,
    save_universal_bundle,
    validate_universal_bundle,
)

OUT_PATH = MODELS_DIR / "motor_imagery_universal.pkl"
REPORT_CSV = OUTPUTS_DIR / "universal_model_validation.csv"


def main() -> None:
    print("=" * 60)
    print("  构建 Universal Smart Router 模型")
    print("=" * 60)

    bundle = build_universal_bundle(MODELS_DIR, default_subject="A08")
    save_universal_bundle(bundle, OUT_PATH)
    print(f"已保存: {OUT_PATH}")

    rows = validate_universal_bundle(bundle, GDF_DIR)
    df = pd.DataFrame(rows)
    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(REPORT_CSV, index=False)

    print("\n各受试者 holdout 准确率（自动路由到对应子模型）:")
    print("-" * 60)
    for _, r in df.iterrows():
        flag = "OK" if r["holdout_accuracy"] >= 0.70 else "!!"
        print(
            f"  {r['subject']} -> {r['routed_to']} | "
            f"holdout={r['holdout_accuracy']:.1%} | CV={r['saved_cv']:.1%} | "
            f"{r['method']} [{flag}]"
        )

    mean_acc = df["holdout_accuracy"].mean()
    n70 = (df["holdout_accuracy"] >= 0.70).sum()
    print("-" * 60)
    print(f"平均 holdout: {mean_acc:.1%}")
    print(f">=70% 受试者: {n70}/{len(df)}")
    print(f"验证报告: {REPORT_CSV}")

    illiterate = df[df["subject"].isin(["A04", "A06"])]
    if not illiterate.empty:
        print(
            "\n注意: A04/A06 为 BCI 失读，单人最优模型也无法达到 70%，"
            "这不是通用模型能解决的。"
        )

    # 可选：把 universal 也设为 default 便于 Streamlit 默认选中
    shutil.copy(OUT_PATH, BEST_MODEL_PATH)
    print(f"\n默认模型已更新为 Universal: {BEST_MODEL_PATH}")


if __name__ == "__main__":
    main()
