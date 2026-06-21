# Pyongyang-VLM

Pyongyang-VLM là MVP video person retrieval / pedestrian search: nhập mô tả bằng ngôn ngữ tự nhiên, tìm người trong video, render bounding box lên track phù hợp và xuất video kết quả.

## Features

- Natural-language person search
- Video person detection, tracking và cropping
- TBPS-CLIP image-text retrieval
- Track-level ranking
- Gradio demo UI
- Bounding box video rendering
- Optional conservative track stitching cho các track bị tách sau occlusion

## Architecture

```text
Raw query
  -> Query Understanding
  -> normalized_text

Video
  -> Vision Pipeline
  -> tracklets / crops / bboxes
  -> video embedding index

normalized_text + video index
  -> Matching Engine
  -> best_track_id + ranking
  -> optional track stitching
  -> renderer
  -> output video
```

Luồng end-to-end hiện tại:

1. User upload hoặc truyền video.
2. Module 2 Vision Pipeline detect/track/crop persons.
3. Demo pipeline build video embedding index bằng TBPS-CLIP image encoder.
4. User nhập query tự nhiên.
5. Module 1 Query Understanding dùng Vertex AI Gemini để normalize query.
6. Module 3 Matching Engine encode text query bằng TBPS-CLIP, tính cosine similarity và rank track candidates.
7. Có thể bật conservative track stitching để nối các `track_id` có khả năng là cùng người.
8. Renderer vẽ bbox lên người được chọn và xuất MP4.

## Requirements

- Python `>=3.12` theo `pyproject.toml`
- `uv` được khuyến nghị để quản lý môi trường
- `ffmpeg` để render video H.264 MP4
- CUDA là optional nhưng rất nên dùng khi export embeddings / chạy TBPS-CLIP
- Google Cloud project có Vertex AI enabled cho Module 1
- TBPS-CLIP checkpoint tại `weights/checkpoint_best.pth`

## Installation

```bash
git clone https://github.com/FWD-LeTung/Pyongyang-VLM.git
cd Pyongyang-VLM
pip install uv
uv sync
```

Nếu máy chưa có `ffmpeg`, cài bằng package manager của hệ điều hành. Ví dụ Ubuntu:

```bash
sudo apt-get update
sudo apt-get install -y ffmpeg
```

## Google Cloud / Vertex AI Setup

Module 1 cần Vertex AI Gemini. Tạo `.env` từ `.env.example` hoặc export trực tiếp các biến sau:

```bash
export GCP_PROJECT_ID="your-project-id"
export GCP_LOCATION="us-central1"
export GEMINI_MODEL="gemini-2.5-flash"
```

Đăng nhập local bằng Application Default Credentials:

```bash
gcloud auth application-default login
```

Trên Colab, đăng nhập bằng `gcloud auth login` và `gcloud auth application-default login`.

Không commit credential, token, service account key hoặc nội dung `.env` thật vào Git.

## Checkpoint Setup

TBPS-CLIP checkpoint cần nằm tại:

```text
weights/checkpoint_best.pth
```

Tải checkpoint tại:

```text
https://drive.google.com/file/d/1Y94znxB7J7UeczulGH9sAzZpScHOhaHT/view?usp=sharing
```

Checkpoint không được commit vào Git. Repo hiện có `weights/yolov8n.pt` cho detector; `weights/checkpoint_best.pth` nằm trong `.gitignore`.

## CLI Demo

Repo có test video tại:

```text
data/test_videos/cctv_full_h264.mp4
```

### 1. Export Video Index

```bash
uv run python demo/export_video_embeddings.py \
  --video data/test_videos/cctv_full_h264.mp4 \
  --output outputs/video_index/cctv_full_h264.pt \
  --max-frames 0 \
  --device cuda \
  --precision fp16
```

`--max-frames 0` nghĩa là xử lý full video. Nếu máy không có CUDA, dùng:

```bash
uv run python demo/export_video_embeddings.py \
  --video data/test_videos/cctv_full_h264.mp4 \
  --output outputs/video_index/cctv_full_h264.pt \
  --max-frames 0 \
  --device cpu \
  --precision fp32
```

### 2. Query Index

```bash
uv run python demo/query_video_embeddings.py \
  --index outputs/video_index/cctv_full_h264.pt \
  --query "viết mô tả bằng tiếng Anh hoặc tiếng Việt" \
  --device cpu \
  --precision fp32 \
  --top-k 10 \
  --save-debug-images
```

Debug images mặc định được lưu vào:

```text
outputs/debug_check/
```

### 3. Render BBox Video

```bash
uv run python demo/render_query_result.py \
  --index outputs/video_index/cctv_full_h264.pt \
  --video data/test_videos/cctv_full_h264.mp4 \
  --query "viết mô tả bằng tiếng Anh hoặc tiếng Việt" \
  --output outputs/rendered_result.mp4 \
  --device cpu \
  --precision fp32 \
  --hold-frames 15 \
  --force-render
```

### 4. Render With Auto Stitching

Auto stitching là query-conditioned và conservative: chỉ nối track khi appearance, temporal gap, overlap và mutual-best checks đủ chắc.

```bash
uv run python demo/render_query_result.py \
  --index outputs/video_index/cctv_full_h264.pt \
  --video data/test_videos/cctv_full_h264.mp4 \
  --query "viết mô tả bằng tiếng Anh hoặc tiếng Việt" \
  --output outputs/rendered_result_stitched.mp4 \
  --device cpu \
  --precision fp32 \
  --hold-frames 15 \
  --force-render \
  --auto-stitch
```

Debug stitch candidates cho một target track:

```bash
uv run python demo/debug_track_stitching.py \
  --index outputs/video_index/cctv_full_h264.pt \
  --target-track-id 261
```

## Gradio UI

Chạy local:

```bash
uv run python app_gradio.py
```

Chạy với public link:

```bash
uv run python app_gradio.py --share
```

UI flow hiện tại:

1. Upload video.
2. Chọn `Precision`.
3. Bấm `Process Video`.
4. Nhập query.
5. Bật/tắt `Auto stitch fragmented tracks` nếu cần.
6. Bấm `Search & Render`.
7. Xem output video trong UI.

Trong code hiện tại, Gradio tự chọn device bằng CUDA nếu có, nếu không thì CPU. Checkpoint mặc định là `weights/checkpoint_best.pth`.

Colab notebook:

```text
https://colab.research.google.com/drive/1oGY7A0diKy6IbBVVXtns7o9jTdm53Hf8?usp=sharing
```

## Configuration

Các file config chính:

- `config/vision_pipeline.yaml`
- `config/matching_engine.yaml`

Một số setting thường cần chỉnh:

- `reader.processing_fps`: FPS xử lý video trong Vision Pipeline
- `detector.device`: device cho detector
- `detector.confidence_threshold`: ngưỡng detect person
- `tracker.track_high_thresh`, `tracker.track_low_thresh`, `tracker.match_thresh`: threshold tracker
- `retrieval.checkpoint_path`: path TBPS-CLIP checkpoint
- `retrieval.device`: device mặc định của Matching Engine
- `retrieval.precision`: `fp16` hoặc `fp32`

CLI args như `--device`, `--precision`, `--checkpoint`, `--vision-config`, `--matching-config` có thể override config trong các demo script.

## Outputs

Các file output thường sinh ra:

- `outputs/video_index/*.pt`: video embedding index
- `outputs/debug_check/`: debug frame/crop images
- `outputs/rendered_result.mp4`: video render bbox
- `outputs/rendered_result_stitched.mp4`: video render có auto stitching
- `outputs/gradio_sessions/`: session data của Gradio UI

`outputs/`, `logs/`, `.env` và `weights/checkpoint_best.pth` đang nằm trong `.gitignore`. Không commit checkpoint, video lớn, output `.pt` hoặc output `.mp4`.

## Troubleshooting

### Vertex AI `LLM_API_ERROR`

Kiểm tra:

- `GCP_PROJECT_ID`, `GCP_LOCATION`, `GEMINI_MODEL`
- Vertex AI API đã enabled
- `gcloud auth application-default login`
- IAM permission của account đang dùng
- Model name còn hợp lệ với region đã chọn

### CUDA unavailable

Dùng CPU:

```bash
--device cpu --precision fp32
```

Hoặc bật GPU runtime / cài CUDA đúng với môi trường.

### `ffmpeg` missing

Renderer cần `ffmpeg` để xuất H.264 MP4. Cài `ffmpeg` trước khi render.

### Video codec issue / AV1

Nếu OpenCV hoặc browser không đọc được video, convert sang H.264:

```bash
ffmpeg -i input.mp4 -c:v libx264 -pix_fmt yuv420p -movflags +faststart output_h264.mp4
```

### Slow Colab UI

Render, upload/download và encode full video có thể chậm. Dùng video ngắn hơn hoặc đặt `--max-frames` nhỏ để test nhanh.

### Tracking starts late hoặc bbox biến mất

Detector/tracker có thể bị ID switch hoặc track fragmentation khi occlusion. Thử `--auto-stitch` hoặc inspect bằng `demo/debug_track_stitching.py`.

## Limitations

- Đây là MVP demo, chưa phải production multi-user system.
- Accuracy phụ thuộc detector/tracker, chất lượng video và TBPS-CLIP checkpoint.
- Track fragmentation vẫn có thể xảy ra khi người bị che khuất hoặc cảnh đông.
- Auto stitching cố tình conservative, nên có thể không nối nếu chưa đủ chắc.
- Full-video embedding export và rendering có thể chậm trên CPU.

## Tests

```bash
uv run python -m compileall src demo app_gradio.py
uv run pytest
```

## Suggested Development Workflow

```bash
git status
uv run python -m compileall src demo app_gradio.py
uv run pytest
```
