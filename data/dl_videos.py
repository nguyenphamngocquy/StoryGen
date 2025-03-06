import os
import json
import subprocess

# Load metadata.json
with open("metadata.json", "r", encoding="utf-8") as file:
    metadata = json.load(file)

# Create the output folder
os.makedirs("video", exist_ok=True)

COOKIES_FILE = "www.youtube.com_cookies.txt"

# Loop through each video entry and download using your exact command
for video_id, video_data in metadata.items():
    video_url = video_data["video_url"][0][0]  # Extract video URL

    # Your exact command with the URL inserted
    command = f'yt-dlp --write-auto-sub --cookies "{COOKIES_FILE}" -o "video/%(title)s.%(ext)s" -f 135 "{video_url}"'
    
    print(f"Downloading from: {video_url}")
    
    try:
        subprocess.run(command, shell=True, check=True)
    except subprocess.CalledProcessError as e:
        print(f"Error downloading {video_url}: {e}")

print("All videos downloaded!")