"""
GDF 训练模块：MNE CSP + SVM（BCI 2a 标准流程）
"""
from __future__ import annotations

import pickle
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from mne.decoding import CSP
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    cohen_kappa_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.svm import SVC

from config import (
    BEST_MODEL_PATH,
    GDF_CSP_COMPONENTS,
    GDF_DIR,
    MODELS_DIR,
    OUTPUTS_DIR,
    RANDOM_STATE,
    TEST_SIZE,
)
from src.fbcsp import FBCSP_LDA
from src.gdf_loader import subject_to_gdf_path
from src.gdf_preprocessing import load_mi_epochs_flexible

BAND_MAP = {
    "8-30Hz": (8.0, 30.0),
    "7-35Hz": (7.0, 35.0),
    "4-40Hz": (4.0, 40.0),
}
WINDOW_MAP = {
    "0.5-2.5s": (0.5, 2.5),
    "0.5-3.5s": (0.5, 3.5),
    "0.5-4.0s": (0.5, 4.0),
    "1.0-4.0s": (1.0, 4.0),
}


def build_csp_svm_pipeline(n_components: int = GDF_CSP_COMPONENTS) -> Pipeline:
    """构建 CSP + 标准化 + SVM 流水线。"""
    return Pipeline(
        [
            ("csp", CSP(n_components=n_components, reg="ledoit_wolf", log=True)),
            ("scaler", StandardScaler()),
            ("svm", SVC(kernel="rbf", random_state=RANDOM_STATE)),
        ]
    )


def evaluate_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """计算四项指标。"""
    return {
        "Accuracy": accuracy_score(y_true, y_pred),
        "Precision": precision_score(y_true, y_pred, average="weighted", zero_division=0),
        "Recall": recall_score(y_true, y_pred, average="weighted", zero_division=0),
        "F1-score": f1_score(y_true, y_pred, average="weighted", zero_division=0),
    }


def plot_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_names: List[str],
    title: str,
    save_path: Path,
) -> None:
    """保存混淆矩阵图。"""
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=class_names, yticklabels=class_names)
    plt.title(title)
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"混淆矩阵已保存: {save_path}")


def build_estimator(method: str = "CSP+SVM", bandpass: tuple = (8.0, 30.0)):
    """构建分类器。"""
    if method == "FBCSP+LDA":
        return FBCSP_LDA(bandpass=bandpass, n_components=2, sample_rate=250.0)
    return build_csp_svm_pipeline()


def load_subject_data(
    subject: str,
    band: str = "8-30Hz",
    window: str = "0.5-2.5s",
    use_car: bool = False,
    gdf_dir: Path | None = None,
):
    """按实验配置加载数据。"""
    gdf_path = subject_to_gdf_path(subject, session="T", gdf_dir=gdf_dir)
    bandpass = BAND_MAP.get(band, (8.0, 30.0))
    tmin, tmax = WINDOW_MAP.get(window, (0.5, 2.5))
    X, y, class_names, _, _ = load_mi_epochs_flexible(
        gdf_path, tmin=tmin, tmax=tmax, bandpass=bandpass, apply_car_ref=use_car
    )
    return X, y, class_names


def save_gdf_model(
    pipeline,
    label_encoder: LabelEncoder,
    class_names: List[str],
    subject: str,
    metrics: Dict[str, float],
    method: str = "CSP+SVM",
    config: Dict[str, Any] | None = None,
    save_path: Path = BEST_MODEL_PATH,
) -> None:
    """保存 GDF 训练模型包。"""
    save_path.parent.mkdir(parents=True, exist_ok=True)
    bundle = {
        "pipeline": pipeline,
        "label_encoder": label_encoder,
        "class_names": class_names,
        "data_source": "gdf",
        "feature_mode": method.lower().replace("+", "_"),
        "subject": subject,
        "model_name": method,
        "metrics": metrics,
        "config": config or {},
    }
    if hasattr(pipeline, "named_steps"):
        bundle["model"] = pipeline.named_steps.get("svm", pipeline)
        bundle["scaler"] = pipeline.named_steps.get("scaler")
        bundle["feature_extractor"] = pipeline.named_steps.get("csp")
    with open(save_path, "wb") as f:
        pickle.dump(bundle, f)
    print(f"模型已保存: {save_path}")


def train_single_subject(
    subject: str,
    gdf_dir: Path | None = None,
    save_model: bool = True,
    model_path: Path = BEST_MODEL_PATH,
    method: str = "CSP+SVM",
    band: str = "8-30Hz",
    window: str = "0.5-2.5s",
    use_car: bool = False,
) -> Dict[str, Any]:
    """训练单个受试者，返回评估结果。"""
    bandpass = BAND_MAP.get(band, (8.0, 30.0))
    X, y, class_names = load_subject_data(subject, band, window, use_car, gdf_dir)

    label_encoder = LabelEncoder()
    label_encoder.fit(class_names)
    y_encoded = y

    pipeline = build_estimator(method, bandpass)

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    cv_scores = cross_val_score(pipeline, X, y_encoded, cv=cv)
    print(f"\n----- {subject} [{method} {band} {window} CAR={use_car}] -----")
    print(f"CV Accuracy: {cv_scores.mean():.4f} (+/- {cv_scores.std():.4f})")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y_encoded, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y_encoded
    )
    pipeline.fit(X_train, y_train)
    y_pred = pipeline.predict(X_test)
    metrics = evaluate_metrics(y_test, y_pred)
    metrics["kappa"] = float(cohen_kappa_score(y_test, y_pred))

    print(classification_report(y_test, y_pred, target_names=class_names, zero_division=0))

    if save_model:
        final_pipeline = build_estimator(method, bandpass)
        final_pipeline.fit(X, y_encoded)
        cm_path = OUTPUTS_DIR / f"confusion_matrix_{subject.lower()}_optimized.png"
        OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
        plot_confusion_matrix(
            y_test, y_pred, class_names,
            f"{method} - {subject}", cm_path,
        )
        save_gdf_model(
            final_pipeline, label_encoder, class_names, subject, metrics,
            method=method,
            config={"band": band, "window": window, "car": use_car},
            save_path=model_path,
        )

    return {"subject": subject, "method": method, "band": band, "window": window,
            "car": use_car, "cv_accuracy": float(cv_scores.mean()),
            "cv_std": float(cv_scores.std()), **metrics}


def train_all_subjects(
    gdf_dir: Path | None = None,
    save_best_as_default: bool = True,
) -> pd.DataFrame:
    """评估全部 9 名受试者，保存汇总表。"""
    results = []
    print("\n========== 全部受试者 CSP + SVM 评估 ==========")
    for i in range(1, 10):
        subject = f"A{i:02d}"
        try:
            result = train_single_subject(
                subject,
                gdf_dir=gdf_dir,
                save_model=True,
                model_path=MODELS_DIR / f"motor_imagery_{subject.lower()}.pkl",
            )
            results.append(result)
            print(f"{subject}: CV={result['cv_accuracy']:.1%}, Test={result['Accuracy']:.1%}\n")
        except FileNotFoundError as exc:
            print(f"跳过 {subject}: {exc}")

    summary = pd.DataFrame(results)
    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
    summary.to_csv(OUTPUTS_DIR / "gdf_subject_summary.csv", index=False)
    print("汇总表:", OUTPUTS_DIR / "gdf_subject_summary.csv")
    print(f"平均 CV 准确率: {summary['cv_accuracy'].mean():.1%}")

    if save_best_as_default and not summary.empty:
        best_row = summary.loc[summary["cv_accuracy"].idxmax()]
        best_subject = best_row["subject"]
        best_path = MODELS_DIR / f"motor_imagery_{best_subject.lower()}.pkl"
        if best_path.exists():
            import shutil

            shutil.copy(best_path, BEST_MODEL_PATH)
            print(f"默认模型已设为 CV 最高受试者 {best_subject}: {BEST_MODEL_PATH}")

    return summary


def train_optimized_all(gdf_dir: Path | None = None) -> pd.DataFrame:
    """按实验得出的每人最优配置训练并保存模型。"""
    config_path = OUTPUTS_DIR / "experiments" / "best_config_per_subject.csv"
    if not config_path.exists():
        raise FileNotFoundError("请先运行 run_experiments_fast.py 和 generate_experiment_report.py")

    best = pd.read_csv(config_path)
    results = []
    print("\n========== 按最优配置训练全部受试者 ==========")
    for _, row in best.iterrows():
        subject = row["subject"]
        result = train_single_subject(
            subject,
            gdf_dir=gdf_dir,
            save_model=True,
            model_path=MODELS_DIR / f"motor_imagery_{subject.lower()}.pkl",
            method=row["method"],
            band=row["band"],
            window=row["window"],
            use_car=bool(row["car"]),
        )
        results.append(result)
        print(f"{subject}: CV={result['cv_accuracy']:.1%}\n")

    summary = pd.DataFrame(results)
    summary.to_csv(OUTPUTS_DIR / "optimized_training_summary.csv", index=False)
    print(f"平均 CV 准确率: {summary['cv_accuracy'].mean():.1%}")

    import shutil
    best_row = summary.loc[summary["cv_accuracy"].idxmax()]
    src = MODELS_DIR / f"motor_imagery_{best_row['subject'].lower()}.pkl"
    if src.exists():
        shutil.copy(src, BEST_MODEL_PATH)
    return summary


def run_gdf_training(subject: Optional[str] = None, gdf_dir: Path | None = None, optimized: bool = False) -> None:
    """GDF 训练入口。"""
    print("=" * 50)
    print("  BCI 2a GDF 训练")
    print("=" * 50)

    folder = gdf_dir or GDF_DIR
    if not folder.exists():
        raise FileNotFoundError(f"GDF 目录不存在: {folder}")

    if optimized:
        summary = train_optimized_all(folder)
        print(f"\n优化训练完成！平均 CV 准确率: {summary['cv_accuracy'].mean():.2%}")
        return

    if subject is None or str(subject).lower() == "all":
        summary = train_all_subjects(folder)
        if not summary.empty:
            best = summary.loc[summary["cv_accuracy"].idxmax()]
            print(f"\n训练完成！最佳受试者 {best['subject']}，CV {best['cv_accuracy']:.2%}")
    else:
        subj = subject if str(subject).upper().startswith("A") else f"A{int(subject):02d}"
        result = train_single_subject(subj, gdf_dir=folder, save_model=True)
        print(f"\n训练完成！{subj} CV {result['cv_accuracy']:.2%}")
