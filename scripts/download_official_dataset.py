#!/usr/bin/env python3
"""Download the public Stack_RL2_range table-wood dataset folder."""
import argparse
from pathlib import Path

FOLDER_URL = "https://drive.google.com/drive/folders/1ZzAVSgeie2886xeIc-EQ88tjTq0ICtIP?usp=sharing"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=Path("data/download"))
    args = parser.parse_args()
    try:
        import gdown
    except ImportError as exc:
        raise SystemExit("Önce scripts/bootstrap_ubuntu.sh çalıştırın") from exc
    args.output.mkdir(parents=True, exist_ok=True)
    gdown.download_folder(FOLDER_URL, output=str(args.output), quiet=False, use_cookies=False)
    print("İndirme dizini:", args.output.resolve())
    print("Çalışma yollarını manifests/datasets.json ile eşleştirip verify_dataset.py çalıştırın.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

