"""
BCI 运动想象分类 - 主训练脚本

运行方式:
    python train.py --source gdf                    # GDF + CSP + SVM（推荐，70%+）
    python train.py --source gdf --subject 1        # 仅训练 A01
    python train.py --source gdf --subject all      # 评估全部 9 名受试者
    python train.py --source gdf --optimized        # 按实验最优配置训练
    python train.py --source csv                    # 旧版 CSV 流程
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

from config import DEFAULT_DATA_PATH, GDF_DIR, PATIENTS_DIR
from src.data_loader import (
    aggregate_by_epoch,
    detect_label_column,
    get_feature_columns,
    inspect_data,
    load_all_patients,
    load_csv,
    load_patient_file,
)
from src.feature_extraction import EEGFeatureExtractor
from src.gdf_trainer import run_gdf_training
from src.preprocessing import (
    check_missing_values,
    encode_labels,
    handle_missing_values,
    normalize_features,
    split_data,
    split_raw_dataframe,
)
from src.train_models import train_and_evaluate_all


def parse_args() -> argparse.Namespace:
    default_source = "gdf" if GDF_DIR.exists() else "csv"

    parser = argparse.ArgumentParser(description="BCI 运动想象分类模型训练")
    parser.add_argument(
        "--source",
        type=str,
        choices=["gdf", "csv"],
        default=default_source,
        help="gdf=原始 GDF（CSP+SVM，准确率高）；csv=CSV 数据",
    )
    parser.add_argument(
        "--subject",
        type=str,
        default="1",
        help="GDF 受试者编号，如 1 / A01；all=全部受试者",
    )
    parser.add_argument(
        "--gdf-dir",
        type=str,
        default=str(GDF_DIR),
        help="GDF 文件目录",
    )
    parser.add_argument(
        "--data",
        type=str,
        default=None,
        help="CSV 模式：单个数据文件路径",
    )
    parser.add_argument(
        "--patients-dir",
        type=str,
        default=str(PATIENTS_DIR),
        help="CSV 模式：分患者文件夹",
    )
    parser.add_argument(
        "--patient",
        type=int,
        default=None,
        help="CSV 模式：patient 编号",
    )
    parser.add_argument(
        "--label",
        type=str,
        default=None,
        help="CSV 模式：标签列名",
    )
    parser.add_argument(
        "--feature-mode",
        type=str,
        choices=["eeg", "csp", "raw"],
        default="eeg",
        help="CSV 模式特征类型",
    )
    parser.add_argument(
        "--use-csp",
        action="store_true",
        help="CSV 模式：追加 CSP 特征",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="CSV 模式：只训练指定模型",
    )
    parser.add_argument(
        "--split-by",
        type=str,
        choices=["epoch", "patient"],
        default="epoch",
        help="CSV 模式：数据划分方式",
    )
    parser.add_argument(
        "--optimized",
        action="store_true",
        help="GDF 模式：按实验最优配置为每人训练模型",
    )
    return parser.parse_args()


def load_training_data(args: argparse.Namespace):
    """CSV 模式加载数据。"""
    if args.data:
        return load_csv(args.data)

    patients_dir = Path(args.patients_dir)
    if args.patient is not None:
        return load_patient_file(args.patient, patients_dir)

    if patients_dir.exists():
        return load_all_patients(patients_dir)

    print(f"未找到 patients 目录，回退到: {DEFAULT_DATA_PATH}")
    return load_csv(str(DEFAULT_DATA_PATH))


def run_csv_training(args: argparse.Namespace) -> None:
    """CSV 数据训练流程。"""
    print("=" * 50)
    print("  BCI 运动想象分类 - CSV 训练")
    print("=" * 50)

    df = load_training_data(args)
    inspect_data(df)
    label_column = detect_label_column(df, args.label)

    if args.feature_mode in ("eeg", "csp"):
        if args.feature_mode == "csp":
            print("\n使用经典 CSP 特征")
        else:
            print("\n使用 EEG 频带功率特征")
            if args.use_csp:
                print("已启用 CSP")

        df_train, df_test = split_raw_dataframe(df, label_column, args.split_by)
        use_csp = args.feature_mode == "csp" or args.use_csp
        extractor = EEGFeatureExtractor(use_csp=use_csp, csp_only=args.feature_mode == "csp")
        if use_csp:
            extractor.fit(df_train)

        X_train = extractor.transform(df_train)
        X_test = extractor.transform(df_test)
        y_train, label_encoder = encode_labels(extractor.extract_labels(df_train))
        y_test = label_encoder.transform(extractor.extract_labels(df_test))
    else:
        feature_cols = get_feature_columns(df, label_column)
        df_agg = aggregate_by_epoch(df, feature_cols, label_column)
        check_missing_values(df_agg, feature_cols)
        df_agg = handle_missing_values(df_agg, feature_cols)
        X = df_agg[feature_cols].values
        y, label_encoder = encode_labels(df_agg[label_column])
        groups = df_agg["patient"].values if "patient" in df_agg.columns else None
        X_train, X_test, y_train, y_test = split_data(X, y, groups, args.split_by)
        extractor = EEGFeatureExtractor(use_csp=False)

    X_train_scaled, X_test_scaled, scaler = normalize_features(X_train, X_test)
    model_names = [args.model] if args.model else None
    if args.feature_mode == "csp" and model_names is None:
        model_names = ["svm"]

    best_name, all_metrics = train_and_evaluate_all(
        X_train_scaled,
        X_test_scaled,
        y_train,
        y_test,
        list(label_encoder.classes_),
        scaler,
        label_encoder,
        extractor,
        args.feature_mode,
        args.split_by,
        model_names,
    )
    print(f"\n训练完成！最佳模型: {best_name}，准确率: {all_metrics[best_name]['Accuracy']:.2%}")


def main() -> None:
    args = parse_args()

    if args.source == "gdf":
        if args.optimized:
            run_gdf_training(gdf_dir=Path(args.gdf_dir), optimized=True)
            return
        subject = None if str(args.subject).lower() == "all" else args.subject
        run_gdf_training(subject=subject, gdf_dir=Path(args.gdf_dir), optimized=False)
        return

    run_csv_training(args)


if __name__ == "__main__":
    main()
