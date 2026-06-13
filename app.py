"""
BCI 运动想象分类 - Streamlit 预测应用

运行方式:
    streamlit run app.py
"""
from __future__ import annotations

import pickle
import re
import sys
from pathlib import Path

import pandas as pd
import streamlit as st
from sklearn.model_selection import train_test_split

PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config import (
    BEST_MODEL_PATH,
    COMMAND_MAPPING,
    DEFAULT_DATA_PATH,
    GDF_DIR,
    LABEL_DISPLAY_NAMES,
    MODELS_DIR,
    PATIENTS_DIR,
    RANDOM_STATE,
    TEST_SIZE,
)
from src.data_loader import aggregate_by_epoch, detect_label_column
from src.feature_extraction import EEGFeatureExtractor
from src.gdf_trainer import BAND_MAP, WINDOW_MAP, load_subject_data
from src.gdf_preprocessing import load_epochs_auto

_SUBJECT_PATTERN = re.compile(r"(A010|A0[1-9])", re.IGNORECASE)


def detect_subject_from_path(gdf_path: Path | str | None) -> str | None:
    if gdf_path is None:
        return None
    match = _SUBJECT_PATTERN.search(Path(gdf_path).stem.upper())
    return match.group(1).upper() if match else None


def resolve_sub_bundle(
    bundle: dict,
    gdf_path: Path | str | None = None,
    subject_override: str | None = None,
) -> tuple[dict, str]:
    if bundle.get("type") != "universal_router":
        return bundle, bundle.get("subject", "A01")
    subjects = bundle.get("subjects", {})
    default = bundle.get("default_subject", "A08")
    if subject_override:
        subj = subject_override.upper().strip()
        if subj.isdigit():
            subj = f"A{int(subj):02d}"
        elif not subj.startswith("A"):
            subj = f"A{subj}"
        if len(subj) == 2 and subj[1].isdigit():
            subj = f"A0{subj[1]}"
        if subj[:3] in subjects:
            return subjects[subj[:3]], subj[:3]
    detected = detect_subject_from_path(gdf_path)
    if detected and detected in subjects:
        return subjects[detected], detected
    return subjects[default], default


@st.cache_resource
def load_model_bundle(model_path: str):
    with open(model_path, "rb") as f:
        return pickle.load(f)


def list_available_models() -> dict[str, Path]:
    """返回 {UNIVERSAL, A01, ..., DEFAULT}。"""
    models: dict[str, Path] = {}
    universal = MODELS_DIR / "motor_imagery_universal.pkl"
    if universal.exists():
        models["UNIVERSAL"] = universal
    for p in sorted(MODELS_DIR.glob("motor_imagery_a*.pkl")):
        subj = p.stem.replace("motor_imagery_", "").upper()
        models[subj] = p
    if BEST_MODEL_PATH.exists():
        models["DEFAULT"] = BEST_MODEL_PATH
    return models


def list_local_gdf_files() -> list[Path]:
    if not GDF_DIR.exists():
        return []
    return sorted(GDF_DIR.glob("A*T.gdf"))


def load_epochs_from_gdf(gdf_path: Path, bundle: dict):
    """按（路由后的）模型包预处理配置加载 GDF。"""
    active_bundle, routed = resolve_sub_bundle(bundle, gdf_path=gdf_path)
    cfg = active_bundle.get("config", {})
    band = cfg.get("band", "8-30Hz")
    window = cfg.get("window", "0.5-2.5s")
    use_car = bool(cfg.get("car", False))
    gdf_format = cfg.get("gdf_format", "auto")
    bandpass = BAND_MAP.get(band, (8.0, 30.0))
    tmin, tmax = WINDOW_MAP.get(window, (0.5, 2.5))
    X, y, class_names, _, _ = load_epochs_auto(
        gdf_path,
        gdf_format=gdf_format,
        tmin=tmin,
        tmax=tmax,
        bandpass=bandpass,
        apply_car_ref=use_car,
    )
    return X, y, class_names, band, window, use_car, active_bundle, routed


def map_to_display_name(raw_label: str) -> str:
    return LABEL_DISPLAY_NAMES.get(raw_label.lower(), raw_label)


def map_to_command(display_name: str) -> str:
    return COMMAND_MAPPING.get(display_name, "Unknown")


def prepare_features_from_upload(df: pd.DataFrame, bundle: dict):
    feature_mode = bundle.get("feature_mode", "raw")
    extractor = bundle.get("feature_extractor")
    feature_cols = bundle.get("feature_cols", [])

    if feature_mode in ("eeg", "csp") and extractor is not None:
        X = extractor.transform(df)
        labels_df = None
        label_col = None
        try:
            label_col = detect_label_column(df)
            raw_labels = extractor.extract_labels(df)
            labels_df = pd.DataFrame({label_col: raw_labels})
        except ValueError:
            pass
        return X, labels_df, label_col, "csv"

    label_col = None
    try:
        label_col = detect_label_column(df)
    except ValueError:
        pass

    if "epoch" in df.columns and "patient" in df.columns:
        if label_col:
            df_features = aggregate_by_epoch(df, feature_cols, label_col)
            labels_df = df_features[[label_col]].copy()
            df_features = df_features[feature_cols]
        else:
            df_features = (
                df.groupby(["patient", "epoch"], as_index=False)[feature_cols].mean()
            )
            labels_df = None
        return df_features.values, labels_df, label_col, "csv"

    available = [c for c in feature_cols if c in df.columns]
    if not available:
        raise ValueError(f"上传文件缺少 EEG 特征列，需要类似: {feature_cols[:3]} ...")
    return df[available].values, None, label_col, "csv"


def run_gdf_demo(bundle: dict, gdf_path: Path) -> None:
    """对指定 GDF 文件运行预测（支持 Universal Smart Router）。"""
    if not gdf_path.exists():
        st.error(f"GDF 不存在: `{gdf_path}`")
        return

    try:
        X, y, class_names, band, window, use_car, active_bundle, routed = load_epochs_from_gdf(
            gdf_path, bundle
        )
    except Exception as exc:
        st.error(f"GDF 加载失败: {exc}")
        return

    pipeline = bundle.get("pipeline")
    label_encoder = bundle["label_encoder"]
    if pipeline is None:
        st.error("模型包中缺少 pipeline，请重新运行 python train.py --source gdf")
        return

    if bundle.get("type") == "universal_router" and hasattr(pipeline, "set_context"):
        pipeline.set_context(gdf_path=gdf_path)
        st.info(f"通用模型已自动路由到 **{routed}**（{active_bundle.get('model_name', '')}）")

    _, X_test, _, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )
    predictions = pipeline.predict(X_test)
    raw_labels = label_encoder.inverse_transform(predictions)
    true_labels = label_encoder.inverse_transform(y_test)

    results = pd.DataFrame(
        {
            "True Label": true_labels,
            "Predicted Label": raw_labels,
            "Motor Imagery Category": [map_to_display_name(x) for x in raw_labels],
            "Control Command": [map_to_command(map_to_display_name(x)) for x in raw_labels],
        }
    )

    accuracy = (predictions == y_test).mean()
    st.metric("测试集准确率", f"{accuracy:.2%}")
    model_name = bundle.get("model_name", "GDF")
    if bundle.get("type") == "universal_router":
        model_subj = routed
        per_cv = bundle.get("per_subject_cv", {}).get(routed)
        cv_note = f" | 该受试者训练 CV ≈ {per_cv:.1%}" if per_cv else ""
    else:
        model_subj = bundle.get("subject", "?")
        cv_note = ""
    st.caption(
        f"模型: **{model_name}** | 路由/受试者: **{model_subj}** | 数据: **{gdf_path.name}** | "
        f"{active_bundle.get('model_name', model_name)} | {band} {window} CAR={use_car} | "
        f"试次: {len(X_test)}{cv_note}"
    )
    if bundle.get("type") != "universal_router" and model_subj not in gdf_path.name.upper():
        st.warning(
            f"当前模型是为 **{model_subj}** 训练的，GDF 为 **{gdf_path.name}**，"
            "跨受试者预测准确率可能下降。建议改用 **UNIVERSAL** 模型。"
        )
    elif bundle.get("type") == "universal_router":
        detected = detect_subject_from_path(gdf_path)
        if detected and detected != routed:
            st.warning(f"文件名识别为 {detected}，但路由到了 {routed}。")
        if routed in ("A04", "A06"):
            st.warning(
                f"**{routed}** 为 BCI 失读受试者，即使用最优单人模型准确率也约 **50%**，"
                "无法达到 70%——这是 EEG 信号限制，不是模型 bug。"
            )
        elif accuracy < 0.70 and routed not in ("A04", "A06"):
            st.caption("该 holdout 略低于 70% 属正常波动；可对比 per_subject_cv。")
    st.dataframe(results.head(20), use_container_width=True)
    st.subheader("最新预测控制命令")
    last_cat = map_to_display_name(raw_labels[-1])
    st.success(f"**{last_cat}** → **{map_to_command(last_cat)}**")


def _run_csv_prediction(df, bundle) -> None:
    model = bundle["model"]
    scaler = bundle["scaler"]
    label_encoder = bundle["label_encoder"]

    X, labels_df, label_col, _ = prepare_features_from_upload(df, bundle)
    X_scaled = scaler.transform(X)
    predictions = model.predict(X_scaled)

    raw_labels = label_encoder.inverse_transform(predictions)
    display_names = [map_to_display_name(lbl) for lbl in raw_labels]
    commands = [map_to_command(name) for name in display_names]

    results = pd.DataFrame(
        {
            "Predicted Label (Raw)": raw_labels,
            "Motor Imagery Category": display_names,
            "Control Command": commands,
        }
    )

    st.subheader("预测结果")
    st.dataframe(results)
    st.subheader("预测分布")
    st.bar_chart(results["Motor Imagery Category"].value_counts())

    if labels_df is not None and label_col:
        true_labels = labels_df[label_col].values[: len(raw_labels)]
        true_encoded = label_encoder.transform(true_labels)
        accuracy = (predictions[: len(true_encoded)] == true_encoded).mean()
        st.metric("预测准确率", f"{accuracy:.2%}")

    st.subheader("最新预测控制命令")
    st.success(f"**{display_names[-1]}** → **{commands[-1]}**")


def main() -> None:
    st.set_page_config(page_title="BCI Motor Imagery Classifier", page_icon="🧠", layout="wide")
    st.title("🧠 BCI 运动想象分类预测")

    available = list_available_models()
    if not available:
        st.error("未找到 models/ 下的 .pkl 模型，请先运行 python train.py --source gdf")
        st.stop()

    st.sidebar.header("设置")

    model_options = list(available.keys())
    default_idx = model_options.index("UNIVERSAL") if "UNIVERSAL" in model_options else (
        model_options.index("A09") if "A09" in model_options else 0
    )
    selected_key = st.sidebar.selectbox(
        "选择模型 / 受试者",
        model_options,
        index=default_idx,
        format_func=lambda k: (
            f"UNIVERSAL 智能通用（自动匹配 A01–A09） ({available[k].name})"
            if k == "UNIVERSAL"
            else f"{k} ({available[k].name})"
        ),
    )
    model_path = available[selected_key]

    st.sidebar.caption(f"路径: `{model_path}`")

    if not model_path.exists():
        st.error(f"未找到模型: `{model_path}`")
        st.stop()

    bundle = load_model_bundle(str(model_path))
    data_source = bundle.get("data_source", "csv")
    model_name = bundle.get("model_name", "Unknown")
    feature_mode = bundle.get("feature_mode", "raw")

    st.sidebar.success(f"模型: **{model_name}**")
    if bundle.get("type") == "universal_router":
        st.sidebar.info("训练受试者: **ALL（自动路由）**")
        cv_map = bundle.get("per_subject_cv", {})
        if cv_map:
            st.sidebar.caption(
                "各受试者 CV: "
                + ", ".join(f"{k} {v:.0%}" for k, v in sorted(cv_map.items()))
            )
    else:
        st.sidebar.info(f"训练受试者: **{bundle.get('subject', '?')}**")
    st.sidebar.markdown("**控制命令:**")
    for display, cmd in COMMAND_MAPPING.items():
        st.sidebar.markdown(f"- {display} → **{cmd}**")

    if data_source == "gdf":
        cfg = bundle.get("config", {})
        cfg_text = ""
        if cfg:
            cfg_text = f"{cfg.get('band', '')} {cfg.get('window', '')} CAR={cfg.get('car', False)}"
        st.markdown(f"**GDF 模式** | 模型: {model_name} | 预处理: {cfg_text}")

        st.subheader("1. 选择 GDF 数据")
        gdf_source = st.radio(
            "数据来源",
            ["本地 BCICIV_2a_gdf 文件夹", "上传 GDF 文件"],
            horizontal=True,
        )

        gdf_path: Path | None = None

        if gdf_source == "本地 BCICIV_2a_gdf 文件夹":
            local_files = list_local_gdf_files()
            if not local_files:
                st.warning(f"文件夹为空或不存在: `{GDF_DIR}`")
            else:
                names = [p.name for p in local_files]
                default_gdf = bundle.get("subject", "A01")
                default_name = f"{default_gdf}T.gdf"
                idx = names.index(default_name) if default_name in names else 0
                picked = st.selectbox("选择 GDF 文件", names, index=idx)
                gdf_path = GDF_DIR / picked
                st.info(f"将使用: `{gdf_path}`")
        else:
            uploaded_gdf = st.file_uploader(
                "上传 .gdf 文件（BCI Competition IV 2a 格式）",
                type=["gdf"],
            )
            if uploaded_gdf is not None:
                import tempfile
                tmp = Path(tempfile.gettempdir()) / f"bci_upload_{uploaded_gdf.name}"
                tmp.write_bytes(uploaded_gdf.getvalue())
                gdf_path = tmp
                st.success(f"已上传: **{uploaded_gdf.name}** ({uploaded_gdf.size // 1024} KB)")

        st.subheader("2. 运行预测")
        st.caption(
            "提示：选 **UNIVERSAL** 时，会根据 GDF 文件名（如 A05T.gdf）自动切换对应最优子模型；"
            "A04/A06 因 BCI 失读上限约 50%。"
        )
        if st.button("运行 GDF 预测", type="primary"):
            if gdf_path is None:
                st.error("请先选择或上传 GDF 文件")
            else:
                with st.spinner("加载 GDF 并推理中..."):
                    run_gdf_demo(bundle, gdf_path)
        return

    st.markdown("上传 CSV 文件进行预测。")
    uploaded_file = st.file_uploader("上传 CSV 文件", type=["csv"])

    if uploaded_file is None:
        demo_choice = st.selectbox("CSV 演示", ["不使用", "Patient 9 单文件"])
        if st.button("运行 CSV 演示") and demo_choice != "不使用":
            df = pd.read_csv(PATIENTS_DIR / "BCICIV_2a_9.csv", nrows=5000)
            _run_csv_prediction(df, bundle)
        return

    df = pd.read_csv(uploaded_file)
    st.dataframe(df.head(10))
    try:
        _run_csv_prediction(df, bundle)
    except Exception as e:
        st.error(f"预测失败: {e}")


if __name__ == "__main__":
    main()
