import argparse
from pathlib import Path

import torch
import cv2
import numpy as np


def check_environment():
    print("=" * 45)
    print("LICENSE PLATE RECOGNITION PROJECT")
    print("=" * 45)

    print(f"PyTorch version: {torch.__version__}")
    print(f"OpenCV version: {cv2.__version__}")
    print(f"NumPy version: {np.__version__}")

    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    print(f"\nSelected device: {device}")

    # Test PyTorch tensor operations
    x = torch.rand(3, 3, device=device)
    y = torch.rand(3, 3, device=device)

    result = torch.matmul(x, y)

    print("\nTensor operation test:")
    print(result)

    print("\nEnvironment is ready!")


def main():
    parser = argparse.ArgumentParser(description="Recunoașterea plăcuțelor dintr-un fișier video.")
    parser.add_argument("--video", type=Path, help="Fișierul video de intrare")
    parser.add_argument("--output", type=Path, help="Fișierul MP4 adnotat")
    checkpoints = Path(__file__).resolve().parent / "models" / "checkpoints"
    parser.add_argument("--detector", type=Path, default=checkpoints / "detector-640.pt")
    parser.add_argument("--ocr", type=Path, default=checkpoints / "ocr-augmentat.pt")
    parser.add_argument("--min-score", type=float, default=0.5, help="Pragul detectorului, între 0 și 1")
    parser.add_argument("--margin", type=float, default=0.30, help="Marja pe fiecare latură, între 0 și 1")
    args = parser.parse_args()
    if args.video is None and args.output is None:
        check_environment()
        return
    if args.video is None or args.output is None:
        parser.error("--video și --output trebuie folosite împreună")
    if not 0 <= args.min_score <= 1 or not 0 <= args.margin <= 1:
        parser.error("--min-score și --margin trebuie să fie între 0 și 1")

    from inference.pipeline import PlateReader, process_video

    try:
        reader = PlateReader(args.detector, args.ocr, args.min_score, args.margin)
        frames, readings = process_video(args.video, args.output, reader)
    except (OSError, ValueError, cv2.error) as error:
        parser.exit(1, f"{error}\n")
    print(f"Cadre procesate: {frames} | Cadre cu predicție OCR: {readings}")
    print(f"Video salvat: {args.output}")


if __name__ == "__main__":
    main()
