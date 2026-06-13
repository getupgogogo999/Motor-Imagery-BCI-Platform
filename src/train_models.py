"""
模型训练与评估模块
- 训练 SVM、Random Forest、XGBoost
- 输出 Accuracy、Precision、Recall、F1-score
- 绘制混淆矩阵并保存最佳模型
"""
from __future__ import annotations

import pickle
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.svm import SVC
from xgboost import XGBClassifier

from config import BEST_MODEL_PATH, MODELS_DIR, OUTPUTS_DIR, RANDOM_STATE


def build_models(model_names: Optional[List[str]] = None) -> Dict[str, Any]:
    """创建待训练的机器学习模型。"""
    all_models = {
        "SVM": SVC(kernel="rbf", random_state=RANDOM_STATE),
        "Random Forest": RandomForestClassifier(
            n_estimators=300,
            random_state=RANDOM_STATE,
            n_jobs=-1,
        ),
        "XGBoost": XGBClassifier(
            n_estimators=300,
            random_state=RANDOM_STATE,
            eval_metric="mlogloss",
            verbosity=0,
            n_jobs=-1,
        ),
    }
    if not model_names:
        return all_models

    selected = {}
    for name in model_names:
        key = name if name in all_models else name.title()
        if key == "Svm":
            key = "SVM"
        if key not in all_models:
            raise ValueError(f"未知模型: {name}，可选: {list(all_models.keys())}")
        selected[key] = all_models[key]
    return selected


def evaluate_model(
    name: str,
    model: Any,
    X_test: np.ndarray,
    y_test: np.ndarray,
) -> Dict[str, float]:
    """计算并打印四项评估指标。"""
    y_pred = model.predict(X_test)

    metrics = {
        "Accuracy": accuracy_score(y_test, y_pred),
        "Precision": precision_score(y_test, y_pred, average="weighted", zero_division=0),
        "Recall": recall_score(y_test, y_pred, average="weighted", zero_division=0),
        "F1-score": f1_score(y_test, y_pred, average="weighted", zero_division=0),
    }

    print(f"\n----- {name} 评估结果 -----")
    for metric_name, value in metrics.items():
        print(f"{metric_name}: {value:.4f}")
    print(classification_report(y_test, y_pred, zero_division=0))

    return metrics


def plot_confusion_matrix(
    y_test: np.ndarray,
    y_pred: np.ndarray,
    class_names: List[str],
    model_name: str,
    save_path: Path,
) -> None:
    """绘制并保存混淆矩阵图。"""
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=class_names,
        yticklabels=class_names,
    )
    plt.title(f"Confusion Matrix - {model_name}")
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"混淆矩阵已保存: {save_path}")


def save_best_model(
    model: Any,
    scaler: Any,
    label_encoder: Any,
    feature_extractor: Any,
    model_name: str,
    feature_mode: str,
    split_by: str,
    save_path: Path = BEST_MODEL_PATH,
) -> None:
    """将最佳模型及预处理组件打包保存。"""
    save_path.parent.mkdir(parents=True, exist_ok=True)
    bundle = {
        "model": model,
        "scaler": scaler,
        "label_encoder": label_encoder,
        "feature_extractor": feature_extractor,
        "feature_mode": feature_mode,
        "split_by": split_by,
        "model_name": model_name,
        "feature_cols": getattr(feature_extractor, "eeg_columns_", []),
    }
    with open(save_path, "wb") as f:
        pickle.dump(bundle, f)
    print(f"\n最佳模型 ({model_name}) 已保存: {save_path}")


def train_and_evaluate_all(
    X_train: np.ndarray,
    X_test: np.ndarray,
    y_train: np.ndarray,
    y_test: np.ndarray,
    class_names: List[str],
    scaler: Any,
    label_encoder: Any,
    feature_extractor: Any,
    feature_mode: str,
    split_by: str,
    model_names: Optional[List[str]] = None,
) -> Tuple[str, Dict[str, Dict[str, float]]]:
    """训练所有模型，比较 F1-score，保存最佳模型和混淆矩阵。"""
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)

    models = build_models(model_names)
    all_metrics: Dict[str, Dict[str, float]] = {}
    best_name = ""
    best_f1 = -1.0
    best_model = None
    best_y_pred = None

    for name, model in models.items():
        print(f"\n========== 正在训练 {name} ==========")
        model.fit(X_train, y_train)
        metrics = evaluate_model(name, model, X_test, y_test)
        all_metrics[name] = metrics

        if metrics["F1-score"] > best_f1:
            best_f1 = metrics["F1-score"]
            best_name = name
            best_model = model
            best_y_pred = model.predict(X_test)

    cm_path = OUTPUTS_DIR / f"confusion_matrix_{best_name.replace(' ', '_').lower()}.png"
    plot_confusion_matrix(y_test, best_y_pred, class_names, best_name, cm_path)

    save_best_model(
        best_model,
        scaler,
        label_encoder,
        feature_extractor,
        best_name,
        feature_mode,
        split_by,
    )

    summary = pd.DataFrame(all_metrics).T
    print("\n========== 模型对比汇总 ==========")
    print(summary.round(4))
    summary.to_csv(OUTPUTS_DIR / "model_comparison.csv")
    print(f"对比结果已保存: {OUTPUTS_DIR / 'model_comparison.csv'}")

    return best_name, all_metrics
