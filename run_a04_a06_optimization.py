"""
A04 / A06 专项优化实验（不重新跑 A01–A09 全网格）

运行:
    python run_a04_a06_optimization.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

from config import GDF_DIR, OUTPUTS_DIR, RANDOM_STATE
from src.a04_a06_experiments import (
    BASELINE_CV,
    SUBJECT_PREPROCESS,
    run_csp_dim_search,
    run_extended_fbcsp,
    run_riemannian,
    run_svm_comparison,
)
from src.experiment_eval import leakage_safe_cv_evaluate
from src.gdf_loader import subject_to_gdf_path
from src.gdf_preprocessing import load_mi_epochs_flexible

TARGET_SUBJECTS = ["A04", "A06"]
OUT_DIR = OUTPUTS_DIR / "experiments" / "a04_a06"
RESULTS_CSV = OUT_DIR / "a04_a06_optimization.csv"
BEST_CSV = OUT_DIR / "a04_a06_best_config.csv"


def load_subject(subject: str, for_extended_fb: bool = False):
    """加载受试者 epoch；扩展 FBCSP 需无全局带通以便子带滤波。"""
    cfg = SUBJECT_PREPROCESS[subject]
    gdf_path = subject_to_gdf_path(subject, session="T", gdf_dir=GDF_DIR)
    bandpass = None if for_extended_fb else cfg["bandpass"]
    X, y, class_names, sfreq, _ = load_mi_epochs_flexible(
        gdf_path,
        tmin=cfg["tmin"],
        tmax=cfg["tmax"],
        bandpass=bandpass,
        apply_car_ref=cfg["car"],
    )
    return X, y, class_names, sfreq


def run_all_for_subject(subject: str) -> list[dict]:
    """对单个受试者运行全部 4 类实验。"""
    print(f"\n{'=' * 50}")
    print(f"  {subject} 专项优化（基线 CV = {BASELINE_CV[subject]:.1%}）")
    print(f"{'=' * 50}")

    rows: list[dict] = []
    cfg = SUBJECT_PREPROCESS[subject]
    print(f"  预处理: band={cfg['bandpass']}, window={cfg['tmin']}-{cfg['tmax']}s, CAR={cfg['car']}")

    # (1) Riemannian — 使用受试者 bandpass 配置
    X, y, _, _ = load_subject(subject, for_extended_fb=False)
    print(f"\n  [1/4] Riemannian Geometry ...")
    r = run_riemannian(subject, X, y)
    rows.append(r)
    print(f"        acc={r['accuracy']:.1%}  Δ={r['improvement_pct']:+.1f}pp")

    # (2) Extended Filter Bank — 无全局带通
    X_fb, y_fb, _, _ = load_subject(subject, for_extended_fb=True)
    print(f"\n  [2/4] Extended Filter Bank ...")
    for clf in ("lda", "svm"):
        r = run_extended_fbcsp(subject, X_fb, y_fb, classifier=clf)
        rows.append(r)
        print(f"        {clf.upper()}: acc={r['accuracy']:.1%}  Δ={r['improvement_pct']:+.1f}pp")

    # (3) CSP dimension search
    print(f"\n  [3/4] CSP n_components search ...")
    for r in run_csp_dim_search(subject, X, y):
        rows.append(r)
        print(f"        n={r['config'].split('=')[1]}: acc={r['accuracy']:.1%}  Δ={r['improvement_pct']:+.1f}pp")

    # (4) SVM-RBF grid vs linear
    print(f"\n  [4/4] SVM RBF grid + linear baseline ...")
    for r in run_svm_comparison(subject, X, y):
        rows.append(r)
        tag = r["config"]
        print(f"        {tag}: acc={r['accuracy']:.1%}  Δ={r['improvement_pct']:+.1f}pp")

    return rows


def summarize_best(df: pd.DataFrame) -> pd.DataFrame:
    """每人最佳配置及提升幅度。"""
    best_rows = []
    for subject in TARGET_SUBJECTS:
        sub = df[df["subject"] == subject]
        idx = sub["accuracy"].idxmax()
        best = sub.loc[idx].copy()
        baseline = BASELINE_CV[subject]
        best["is_best"] = True
        best["target_60_reached"] = best["accuracy"] >= 0.60
        best["target_65_reached"] = best["accuracy"] >= 0.65
        best_rows.append(best)

        status = "成功 (≥65%)" if best["accuracy"] >= 0.65 else (
            "达标 (≥60%)" if best["accuracy"] >= 0.60 else "未达 60%"
        )
        print(f"\n  {subject} 最佳: [{best['module']}] {best['config']}")
        print(f"    CV={best['accuracy']:.1%}  基线={baseline:.1%}  提升={best['improvement_pct']:+.1f}pp  → {status}")

    return pd.DataFrame(best_rows)


def main() -> None:
    try:
        import pyriemann  # noqa: F401
    except ImportError:
        print("正在安装 pyriemann ...")
        import subprocess
        subprocess.check_call([sys.executable, "-m", "pip", "install", "pyriemann", "-q"])

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    all_rows: list[dict] = []

    for subject in TARGET_SUBJECTS:
        all_rows.extend(run_all_for_subject(subject))

    df = pd.DataFrame(all_rows)
    df = df.sort_values(["subject", "accuracy"], ascending=[True, False])
    df.to_csv(RESULTS_CSV, index=False)

    print(f"\n{'=' * 50}")
    print("  汇总")
    print(f"{'=' * 50}")
    print(f"  共 {len(df)} 组实验结果 → {RESULTS_CSV}")

    best_df = summarize_best(df)
    best_df.to_csv(BEST_CSV, index=False)
    print(f"  最佳配置 → {BEST_CSV}")

    # 按模块平均
    print("\n  各模块平均 Accuracy (A04+A06):")
    mod_mean = df.groupby("module")["accuracy"].mean().sort_values(ascending=False)
    for mod, acc in mod_mean.items():
        print(f"    {mod}: {acc:.1%}")


if __name__ == "__main__":
    main()
