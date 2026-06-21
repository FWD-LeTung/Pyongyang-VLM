"""Export sampled Module 2 person-crop embeddings to a temporary .pt index."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.demo_pipeline.video_indexing import (
    build_video_index,
    print_summary,
    resolve_device,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--video", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument(
        "--max-frames",
        type=int,
        default=0,
        help="Maximum processed frames. Use 0 for the full video.",
    )
    parser.add_argument("--vision-config", default="config/vision_pipeline.yaml")
    parser.add_argument("--matching-config", default="config/matching_engine.yaml")
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument("--precision", default=None, choices=["fp32", "fp16"])
    parser.add_argument("--session-id", default="export")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    device = resolve_device(args.device)
    precision = args.precision or ("fp16" if device.startswith("cuda") else "fp32")
    max_frames = None if args.max_frames == 0 else args.max_frames

    if device == "cpu":
        print("WARNING: running image export on CPU; TBPS-CLIP encoding may be slow.")

    print("Running Module 2 and encoding sampled crops...")
    index = build_video_index(
        video_path=args.video,
        vision_config=args.vision_config,
        matching_config=args.matching_config,
        checkpoint=args.checkpoint,
        device=device,
        precision=precision,
        max_frames=max_frames,
        session_id=args.session_id,
    )

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(index, output_path)
    print_summary(index, output_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
