import cv2

def check_video_specs(video_path):
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        print("Error: Could not open video.")
        return 
    
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)

    print(f"Video Specifications:")
    print(f"  Width: {width}")
    print(f"  Height: {height}")
    print(f"  FPS: {fps}")

    cap.release()

if __name__ == "__main__":
    video_path = "data/test_videos/cctv_full.mp4"
    check_video_specs(video_path)