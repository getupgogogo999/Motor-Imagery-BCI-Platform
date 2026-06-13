"""
下载/准备外部受试者 A010（BCI Competition IV 2b 左右手 MI）

来源: B0101T.gdf（受试者 B01 第 1 次训练 session）
      与 A01-A09 不同实验室受试者、3 通道 EEG、仅左右手 2 类

用法:
    python run_prepare_a010_external.py
"""
from __future__ import annotations

import shutil
import sys
from pathlib import Path

import numpy as np
from mne.decoding import CSP
from sklearn.metrics import accuracy_score, cohen_kappa_score
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.svm import SVC

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

from config import GDF_DIR, MODELS_DIR, OUTPUTS_DIR, RANDOM_STATE, TEST_SIZE
from src.gdf_preprocessing import load_mi_epochs_lr_flexible
from src.gdf_trainer import evaluate_metrics, save_gdf_model

BCI2B_DIR = PROJECT_ROOT / "BCICIV_2b_gdf"
SOURCE_GDF = BCI2B_DIR / "B0101T.gdf"
A010_GDF = GDF_DIR / "A010T.gdf"
DEMO_GDF = PROJECT_ROOT / "demo_samples" / "A010T.gdf"
MODEL_PATH = MODELS_DIR / "motor_imagery_a010.pkl"


def build_a010_pipeline(n_channels: int) -> Pipeline:
    n_csp = min(3, max(2, n_channels))
    return Pipeline([
        ("csp", CSP(n_components=n_csp, reg="ledoit_wolf", log=True)),
        ("scaler", StandardScaler()),
        ("svm", SVC(kernel="rbf", random_state=RANDOM_STATE)),
    ])


def main() -> None:
    if not SOURCE_GDF.exists():
        raise FileNotFoundError(
            f"未找到 {SOURCE_GDF}，请先下载 BCI IV 2b:\n"
            "https://www.bbci.de/competition/download/competition_iv/BCICIV_2b_gdf.zip"
        )

    GDF_DIR.mkdir(parents=True, exist_ok=True)
    DEMO_GDF.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(SOURCE_GDF, A010_GDF)
    shutil.copy2(SOURCE_GDF, DEMO_GDF)
    print(f"已复制 -> {A010_GDF}")
    print(f"已复制 -> {DEMO_GDF}")

    X, y, class_names, sfreq, ch_names = load_mi_epochs_lr_flexible(
        A010_GDF, tmin=0.5, tmax=2.5, bandpass=(8.0, 30.0), apply_car_ref=False, verbose=True
    )
    print(f"通道: {ch_names}  采样率: {sfreq}Hz  试次: {len(y)}")

    pipeline = build_a010_pipeline(X.shape[1])
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    cv_scores = cross_val_score(pipeline, X, y, cv=cv)
    print(f"5-fold CV: {cv_scores.mean():.1%} (+/- {cv_scores.std():.1%})")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )
    pipeline.fit(X_train, y_train)
    y_pred = pipeline.predict(X_test)
    metrics = evaluate_metrics(y_test, y_pred)
    metrics["kappa"] = float(cohen_kappa_score(y_test, y_pred))
    print(f"Holdout: {metrics['Accuracy']:.1%}")

    label_encoder = LabelEncoder()
    label_encoder.fit(class_names)
    final = build_a010_pipeline(X.shape[1])
    final.fit(X, y)

    save_gdf_model(
        final,
        label_encoder,
        class_names,
        subject="A010",
        metrics=metrics,
        method="CSP+SVM (BCI2b LR)",
        config={
            "band": "8-30Hz",
            "window": "0.5-2.5s",
            "car": False,
            "gdf_format": "bci2b_lr",
            "source_dataset": "BCI Competition IV 2b",
            "source_file": "B0101T.gdf",
            "source_subject": "B01",
            "n_channels": len(ch_names),
            "classes": "left,right only",
        },
        save_path=MODEL_PATH,
    )
    print(f"模型已保存: {MODEL_PATH}")

    readme = DEMO_GDF.parent / "A010_README.txt"
    readme.write_text(
        "A010T.gdf — 外部测试受试者\n"
        "========================\n\n"
        "来源: BCI Competition IV Dataset 2b, 受试者 B01, session 1 (B0101T.gdf)\n"
        "官网: https://www.bbci.de/competition/iv/#dataset2b\n\n"
        "与 A01-A09 (2a) 的区别:\n"
        "  - 仅左右手 2 类 (769/770)，无脚/舌\n"
        "  - 3 通道 EEG (C3, Cz, C4)，非 22 通道\n"
        "  - 全新受试者，用于测试 Universal 路由\n\n"
        "Streamlit: 选 UNIVERSAL + A010T.gdf\n",
        encoding="utf-8",
    )

    summary = OUTPUTS_DIR / "a010_external_summary.txt"
    summary.write_text(
        f"A010 external subject (BCI 2b B01)\n"
        f"CV accuracy: {cv_scores.mean():.4f}\n"
        f"Holdout: {metrics['Accuracy']:.4f}\n"
        f"Trials: {len(y)} (left/right各约60)\n",
        encoding="utf-8",
    )

    print("\n下一步: python run_build_universal_model.py  # 把 A010 并入 Universal")


if __name__ == "__main__":
    main()
