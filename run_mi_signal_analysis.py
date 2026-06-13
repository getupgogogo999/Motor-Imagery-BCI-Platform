"""
MI 信号诊断：ERD/ERS + t-SNE（验证 BCI illiteracy）

运行:
    python run_mi_signal_analysis.py
    python run_mi_signal_analysis.py --subjects A03 A04 A06
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.mi_signal_diagnostics import run_comparison


def main():
    parser = argparse.ArgumentParser(description="MI 信号 ERD/ERS + t-SNE 诊断")
    parser.add_argument(
        "--subjects",
        nargs="+",
        default=["A03", "A04", "A06"],
        help="受试者列表（默认 A03 对照 + A04/A06 低表现）",
    )
    args = parser.parse_args()
    run_comparison(subjects=args.subjects)


if __name__ == "__main__":
    main()
