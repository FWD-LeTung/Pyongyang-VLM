import yt_dlp

YOUTUBE_URL = 'https://www.youtube.com/watch?v=YzcawvDGe4Y'

ydl_opts ={
    'format': 'bestvideo[height<=1080][ext=mp4]+bestaudio[ext=m4a]/best[ext=mp4]/best',
    'outtmpl': 'data/test_videos/cctv_full.%(ext)s', 
}

print("Downloading video from YouTube...")
with yt_dlp.YoutubeDL(ydl_opts) as ydl:
    ydl.download([YOUTUBE_URL])
print("Download completed.")