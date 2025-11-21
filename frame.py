import cv2
import os

# Path video tunggal
video_path = r"D:\goldfish-yolov8\videos\mas7.mp4"

# Folder output
output_folder = r"D:\goldfish-yolov8\frame"
os.makedirs(output_folder, exist_ok=True)

frame_interval = 10

cap = cv2.VideoCapture(video_path)
video_name = os.path.splitext(os.path.basename(video_path))[0]

frame_count = 0
saved_count = 0

while True:
    ret, frame = cap.read()
    if not ret:
        break

    if frame_count % frame_interval == 0:
        frame_filename = f"{video_name}_frame_{saved_count}.jpg"
        frame_path = os.path.join(output_folder, frame_filename)
        cv2.imwrite(frame_path, frame)
        saved_count += 1

    frame_count += 1

cap.release()
print(f"✅ {saved_count} frames disimpan dari {video_name}")
print("🎉 Selesai ekstraksi video!")
