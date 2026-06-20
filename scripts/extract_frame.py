import cv2
from pathlib import Path

video_path = "data/test_videos/cctv_full_h264.mp4"
frame_id = 2094
bbox = [1045, 315, 1235, 951]

cap = cv2.VideoCapture(video_path)
cap.set(cv2.CAP_PROP_POS_FRAMES, frame_id)
ok, frame = cap.read()
cap.release()

if not ok:
    raise RuntimeError("Cannot read frame")

x1, y1, x2, y2 = bbox
crop = frame[y1:y2, x1:x2]

Path("outputs/debug_check").mkdir(parents=True, exist_ok=True)
cv2.imwrite("outputs/debug_check/frame_2094.jpg", frame)
cv2.imwrite("outputs/debug_check/track_258_crop.jpg", crop)