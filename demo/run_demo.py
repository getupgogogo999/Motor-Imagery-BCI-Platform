"""
一键 BCI Demo

用法:
    python demo/run_demo.py                  # 游戏内菜单选模型 + GDF
    python demo/run_demo.py --subject A09 --no-prompt
    python demo/run_demo.py --gdf BCICIV_2a_gdf/A03T.gdf --subject A09 --no-prompt
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from config import GDF_DIR, MODELS_DIR
from bci_platform.inference_engine import InferenceEngine
from bci_platform.game.ball_game import run_ball_game
from bci_platform.startup_prompt import prompt_startup


def resolve_model(subject: str | None) -> Path:
    subj = (subject or "A09").upper()
    if not subj.startswith("A"):
        subj = f"A{int(subject):02d}"
    path = MODELS_DIR / f"motor_imagery_{subj.lower()}.pkl"
    if not path.exists():
        raise FileNotFoundError(f"未找到模型: {path}")
    return path


def resolve_gdf(subject: str, gdf_arg: str | None) -> Path:
    if gdf_arg:
        p = Path(gdf_arg)
        if not p.exists():
            raise FileNotFoundError(f"GDF 不存在: {p}")
        return p
    subj = subject.upper()
    if not subj.startswith("A"):
        subj = f"A{int(subject):02d}"
    default = GDF_DIR / f"{subj}T.gdf"
    if not default.exists():
        raise FileNotFoundError(f"GDF 不存在: {default}")
    return default


def main():
    parser = argparse.ArgumentParser(description="BCI 推理 Demo")
    parser.add_argument("--subject", type=str, default=None)
    parser.add_argument("--model", type=str, default=None)
    parser.add_argument("--gdf", type=str, default=None)
    parser.add_argument("--no-prompt", action="store_true", help="跳过菜单，用命令行参数")
    parser.add_argument("--cli-prompt", action="store_true", help="用终端文字菜单（非 pygame）")
    parser.add_argument("--replay", action="store_true")
    parser.add_argument("--api", action="store_true")
    args = parser.parse_args()

    if args.api:
        model_path = Path(args.model) if args.model else resolve_model(args.subject or "A09")
        from bci_platform.api.server import main as api_main
        sys.argv = ["server", "--model", str(model_path)]
        api_main()
        return

    if args.replay:
        subject = args.subject or "A09"
        model_path = Path(args.model) if args.model else resolve_model(subject)
        engine = InferenceEngine(model_path=model_path)
        stats = engine.replay_all(batch_size=32)
        print(f"Replay: {stats['subject']} | {stats['n_trials']} trials | Acc={stats['accuracy']:.1%}")
        return

    if args.cli_prompt:
        _, model_path, gdf_path = prompt_startup()
        run_ball_game(model_path=str(model_path), gdf_path=str(gdf_path))
        return

    if args.no_prompt and (args.subject or args.model or args.gdf):
        subject = args.subject or "A09"
        model_path = Path(args.model) if args.model else resolve_model(subject)
        gdf_path = Path(args.gdf) if args.gdf else resolve_gdf(subject, None)
        run_ball_game(model_path=str(model_path), gdf_path=str(gdf_path))
        return

    # 默认：pygame 启动菜单（选模型 + GDF）
    run_ball_game(show_menu=True)


if __name__ == "__main__":
    main()
