"""Gradio MVP for video person retrieval with required bbox rendering."""

from __future__ import annotations

import argparse
import shutil
import sys
import uuid
from pathlib import Path
from typing import Any

import gradio as gr
import torch

PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.demo_pipeline.query_index import query_video_index
from src.demo_pipeline.video_indexing import build_video_index, resolve_device
from src.demo_pipeline.video_renderer import get_track_timeline, render_track_video


DEFAULT_VISION_CONFIG = "config/vision_pipeline.yaml"
DEFAULT_MATCHING_CONFIG = "config/matching_engine.yaml"
DEFAULT_CHECKPOINT = "weights/checkpoint_best.pth"
DEFAULT_MAX_FRAMES = 0
DEFAULT_SCORE_TOPK = 5
DEFAULT_HOLD_FRAMES = 15
SESSION_ROOT = Path("outputs/gradio_sessions")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--server-name", default="0.0.0.0")
    parser.add_argument("--server-port", type=int, default=7860)
    parser.add_argument("--share", action="store_true")
    return parser.parse_args()


def default_device() -> str:
    return "0" if torch.cuda.is_available() else "cpu"


def default_precision(device: str) -> str:
    return "fp16" if device == "0" or device.startswith("cuda") else "fp32"


def process_video(
    uploaded_video: Any,
    precision: str,
) -> tuple[str, dict[str, Any] | None, str | None, str | None]:
    if uploaded_video is None:
        return "Please upload a video first.", None, None, None

    try:
        session_id = uuid.uuid4().hex
        session_dir = SESSION_ROOT / session_id
        session_dir.mkdir(parents=True, exist_ok=True)
        video_path = copy_uploaded_video(uploaded_video, session_dir)
        resolved_device = resolve_device(default_device())
        resolved_precision = precision or default_precision(resolved_device)

        index_data = build_video_index(
            video_path=video_path,
            vision_config=DEFAULT_VISION_CONFIG,
            matching_config=DEFAULT_MATCHING_CONFIG,
            checkpoint=DEFAULT_CHECKPOINT,
            device=resolved_device,
            precision=resolved_precision,
            max_frames=None if DEFAULT_MAX_FRAMES == 0 else DEFAULT_MAX_FRAMES,
            session_id=session_id,
        )
    except Exception as exc:
        raise gr.Error(str(exc)) from exc

    return (
        format_index_status(index_data),
        index_data,
        str(video_path),
        str(session_dir),
    )


def search_and_render(
    query: str,
    precision: str,
    index_data: dict[str, Any] | None,
    video_path: str | None,
    session_dir: str | None,
) -> tuple[str, str | None]:
    if index_data is None:
        return "Please process a video first.", None
    if not query or not query.strip():
        return "Please enter a query.", None

    try:
        resolved_device = resolve_device(default_device())
        resolved_precision = precision or default_precision(resolved_device)

        result = query_video_index(
            index_data=index_data,
            raw_query=query.strip(),
            matching_config=DEFAULT_MATCHING_CONFIG,
            checkpoint=DEFAULT_CHECKPOINT,
            device=resolved_device,
            precision=resolved_precision,
            score_topk=DEFAULT_SCORE_TOPK,
        )

        best_track_id = result["best_track_id"]
        if best_track_id is None:
            return "No ranked track was produced by query_video_index().", None

        best_score = float(result["best_score"])

        raw_video_path = video_path or index_data.get("video_path")
        if not raw_video_path:
            raise RuntimeError("No video path is available for rendering.")
        source_video = Path(raw_video_path)
        output_dir = Path(session_dir) if session_dir else SESSION_ROOT / uuid.uuid4().hex
        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = output_dir / "rendered_result.mp4"

        timeline = get_track_timeline(index_data, int(best_track_id))
        rendered_path = render_track_video(
            video_path=source_video,
            output_path=output_path,
            track_id=int(best_track_id),
            timeline=timeline,
            score=best_score,
            hold_frames=DEFAULT_HOLD_FRAMES,
        )
    except Exception as exc:
        raise gr.Error(str(exc)) from exc

    return (
        format_query_summary(
            result=result,
            best_track_id=int(best_track_id),
            decision="rendered",
            output_path=rendered_path,
        ),
        str(rendered_path),
    )


def copy_uploaded_video(uploaded_video: Any, session_dir: Path) -> Path:
    source_path = uploaded_file_path(uploaded_video)
    if not source_path.exists():
        raise FileNotFoundError(f"Uploaded video was not found: {source_path}")

    suffix = source_path.suffix or ".mp4"
    target_path = session_dir / f"input{suffix}"
    shutil.copy2(source_path, target_path)
    return target_path


def uploaded_file_path(uploaded_file: Any) -> Path:
    if isinstance(uploaded_file, str | Path):
        return Path(uploaded_file)
    if isinstance(uploaded_file, dict):
        raw_path = uploaded_file.get("path") or uploaded_file.get("name")
        if raw_path:
            return Path(raw_path)
    raw_name = getattr(uploaded_file, "name", None)
    if raw_name:
        return Path(raw_name)
    raise ValueError("Unsupported uploaded video value from Gradio.")


def format_index_status(index_data: dict[str, Any]) -> str:
    stats = index_data.get("stats", {})
    config = index_data.get("config", {})
    return "\n".join(
        [
            "### Index Ready",
            "",
            f"- video path: `{index_data.get('video_path')}`",
            f"- processed_frames: `{stats.get('processed_frames')}`",
            f"- read_frames: `{stats.get('read_frames')}`",
            f"- skipped_frames: `{stats.get('skipped_frames')}`",
            f"- payload_chunks: `{stats.get('num_payload_chunks')}`",
            f"- candidates: `{stats.get('num_candidates')}`",
            f"- embeddings: `{stats.get('num_embeddings')}`",
            f"- embedding_dim: `{stats.get('embedding_dim')}`",
            f"- device: `{config.get('device')}`",
            f"- precision: `{config.get('precision')}`",
        ]
    )


def format_query_summary(
    *,
    result: dict[str, Any],
    best_track_id: int,
    decision: str,
    output_path: Path,
) -> str:
    query_payload = result["query_payload"]
    warnings = result.get("warnings", [])
    lines = [
        "### Query",
        "",
        f"- original_query: `{query_payload.metadata.original_query}`",
        "- normalized_text: "
        f"`{query_payload.vector_search_payload.normalized_text}`",
        "",
        "### Best Match",
        "",
        f"- best_track_id: `{best_track_id}`",
        f"- best_score: `{float(result['best_score']):.6f}`",
        f"- top1_top2_margin: `{float(result['top1_top2_margin']):.6f}`",
        f"- decision: `{decision}`",
        "",
        "### Render",
        "",
        f"- output: `{output_path}`",
        f"- hold_frames: `{DEFAULT_HOLD_FRAMES}`",
    ]
    if warnings:
        lines.extend(["", "### Warnings", ""])
        lines.extend(f"- {warning}" for warning in warnings)
    return "\n".join(lines)


def create_demo() -> gr.Blocks:
    initial_device = default_device()
    initial_precision = default_precision(initial_device)

    with gr.Blocks(title="Video Person Retrieval") as demo:
        gr.Markdown("# Video Person Retrieval")

        index_data_state = gr.State(None)
        video_path_state = gr.State(None)
        session_dir_state = gr.State(None)

        with gr.Tab("1. Process Video"):
            video_input = gr.File(
                label="Video",
                file_types=[".mp4", ".mov", ".avi", ".mkv", ".webm"],
                type="filepath",
            )
            precision_input = gr.Dropdown(
                label="Precision",
                choices=["fp32", "fp16"],
                value=initial_precision,
            )
            process_button = gr.Button("Process Video", variant="primary")
            index_status = gr.Markdown()

        with gr.Tab("2. Search & Render"):
            query_input = gr.Textbox(
                label="Query",
                lines=3,
                placeholder="Describe the person to retrieve.",
            )
            search_button = gr.Button("Search & Render", variant="primary")
            result_summary = gr.Markdown()
            output_video = gr.Video(label="Rendered video")

        process_button.click(
            fn=process_video,
            inputs=[
                video_input,
                precision_input,
            ],
            outputs=[
                index_status,
                index_data_state,
                video_path_state,
                session_dir_state,
            ],
        )
        search_button.click(
            fn=search_and_render,
            inputs=[
                query_input,
                precision_input,
                index_data_state,
                video_path_state,
                session_dir_state,
            ],
            outputs=[result_summary, output_video],
        )

    return demo


def main() -> None:
    args = parse_args()
    demo = create_demo()
    demo.queue()
    demo.launch(
        server_name=args.server_name,
        server_port=args.server_port,
        share=args.share,
    )


if __name__ == "__main__":
    main()
