"""启动时交互选择受试者与 GDF 数据文件。"""
from __future__ import annotations

from pathlib import Path
from typing import Optional, Tuple

from config import GDF_DIR, MODELS_DIR
from bci_platform.model_registry import ModelRegistry


def list_gdf_files(gdf_dir: Path | None = None) -> list[Path]:
    folder = gdf_dir or GDF_DIR
    if not folder.exists():
        return []
    return sorted(folder.glob("A*T.gdf"))


def list_model_subjects() -> list[str]:
    subs = []
    universal = MODELS_DIR / "motor_imagery_universal.pkl"
    if universal.exists():
        subs.append("UNIVERSAL")
    for p in sorted(MODELS_DIR.glob("motor_imagery_a*.pkl")):
        subs.append(p.stem.replace("motor_imagery_", "").upper())
    return subs


def prompt_startup() -> Tuple[str, Path, Optional[Path]]:
    """
    终端交互：选择模型受试者 + GDF 文件。
    返回 (subject, model_path, gdf_path)
    """
    print("\n" + "=" * 50)
    print("  BCI 小球游戏 — 启动配置")
    print("=" * 50)

    models = list_model_subjects()
    if not models:
        raise FileNotFoundError(f"未找到模型，请先训练并保存到 {MODELS_DIR}")

    print("\n【1】选择模型 / 受试者（用于推理 pipeline）:\n")
    for i, subj in enumerate(models, 1):
        p = MODELS_DIR / f"motor_imagery_{subj.lower()}.pkl"
        print(f"  {i}. {subj}  ({p.name})")
    print(f"  0. 使用默认 motor_imagery_model.pkl")

    while True:
        choice = input("\n请输入编号 [默认 9=A09]: ").strip()
        if choice == "":
            subject = "A09" if "A09" in models else models[-1]
            break
        if choice == "0":
            model_path = MODELS_DIR / "motor_imagery_model.pkl"
            if model_path.exists():
                bundle = ModelRegistry.get().load(model_path)
                subject = bundle.get("subject", "A03")
                break
            print("  默认模型不存在，请选其他编号。")
            continue
        try:
            idx = int(choice)
            if 1 <= idx <= len(models):
                subject = models[idx - 1]
                break
        except ValueError:
            if choice.upper() in models:
                subject = choice.upper()
                break
        print("  无效输入，请重试。")

    model_path = MODELS_DIR / f"motor_imagery_{subject.lower()}.pkl"
    if not model_path.exists():
        model_path = MODELS_DIR / "motor_imagery_model.pkl"

    gdf_files = list_gdf_files()
    default_gdf = GDF_DIR / f"{subject}T.gdf"
    print(f"\n【2】选择 GDF 数据文件（replay 试次来源）:\n")
    print(f"  1. 默认（与受试者匹配）: {default_gdf.name}")

    for i, gf in enumerate(gdf_files[:12], 2):
        mark = " ← 推荐" if gf.name == default_gdf.name else ""
        print(f"  {i}. {gf.name}{mark}")
    print(f"  c. 自定义路径")

    gdf_path: Optional[Path] = None
    while True:
        gchoice = input(f"\n请输入编号 [默认 1]: ").strip()
        if gchoice in ("", "1"):
            gdf_path = default_gdf if default_gdf.exists() else (gdf_files[0] if gdf_files else None)
            break
        if gchoice.lower() == "c":
            custom = input("  GDF 完整路径: ").strip().strip('"')
            gdf_path = Path(custom)
            if gdf_path.exists():
                break
            print("  文件不存在，请重试。")
            continue
        try:
            idx = int(gchoice)
            if idx == 1:
                gdf_path = default_gdf
                break
            if 2 <= idx < 2 + len(gdf_files):
                gdf_path = gdf_files[idx - 2]
                break
        except ValueError:
            pass
        print("  无效输入，请重试。")

    if gdf_path is None or not gdf_path.exists():
        raise FileNotFoundError(f"GDF 不存在: {gdf_path}")

    print(f"\n已选择:")
    print(f"  模型: {model_path}")
    print(f"  数据: {gdf_path}")
    print(f"  操作: 空格 = 下一条 trial  |  ESC = 退出  |  R = 重置\n")

    return subject, model_path, gdf_path
