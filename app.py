import os
import uuid
import time
from datetime import datetime

import cv2
import numpy as np
import pandas as pd
from flask import (
    Flask,
    render_template,
    request,
    jsonify,
    send_from_directory,
    url_for,
)
from ultralytics import YOLO
from norfair import Detection, Tracker

# =================================================================
# Custom Euclidean Distance (karena norfair==2.2.0 tidak punya lagi)
# =================================================================
def euclidean_distance(points1, points2):
    """
    Distance function untuk Norfair Tracking.
    points1 dan points2 = array Nx2.
    """
    return np.linalg.norm(points1 - points2, axis=1)


# =================================================================
# Konfigurasi dasar folder
# =================================================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

UPLOAD_DIR = os.path.join(BASE_DIR, "uploads")
OUTPUT_DIR = os.path.join(BASE_DIR, "outputs")
MODEL_PATH = os.path.join(BASE_DIR, "models", "best.pt")

os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Kalibrasi piksel ke cm (silakan sesuaikan)
PX_PER_CM = 25.0


# =================================================================
# Load model YOLO Pose
# =================================================================
print(f"[INFO] Loading YOLO Pose model from: {MODEL_PATH}")
model = YOLO(MODEL_PATH)


# =================================================================
# Tracker Norfair
# =================================================================
tracker = Tracker(
    distance_function=euclidean_distance,
    distance_threshold=30,
)


app = Flask(__name__)


def generate_run_id():
    """Membuat ID unik untuk setiap proses video."""
    return datetime.now().strftime("%Y%m%d-%H%M%S") + "-" + uuid.uuid4().hex[:6]


# =================================================================
# PROSES VIDEO
# =================================================================
def process_video(input_path: str):
    run_id = generate_run_id()
    print(f"[INFO] Start processing video, run_id={run_id}")

    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        raise RuntimeError("Video tidak dapat dibuka.")

    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0:
        fps = 25.0

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    output_video_name = f"{run_id}_annotated.mp4"
    output_csv_name = f"{run_id}_summary.csv"

    output_video_path = os.path.join(OUTPUT_DIR, output_video_name)
    output_csv_path = os.path.join(OUTPUT_DIR, output_csv_name)

    out = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))

    records = []
    frame_idx = 0
    total_detections = 0
    t0 = time.time()

    global tracker
    tracker = Tracker(
        distance_function=euclidean_distance,
        distance_threshold=30,
    )

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_idx += 1
        results = model(frame, verbose=False)
        res = results[0]

        detections = []

        if res.keypoints is not None:
            kpts = res.keypoints.xy.cpu().numpy()
            num_instances = kpts.shape[0]

            boxes = None
            confs = None
            if res.boxes is not None and len(res.boxes) == num_instances:
                boxes = res.boxes.xyxy.cpu().numpy()
                confs = res.boxes.conf.cpu().numpy()

            for i in range(num_instances):
                head = kpts[i, 0]
                tail = kpts[i, 1]

                length_px = np.linalg.norm(head - tail)
                length_cm = length_px / PX_PER_CM if PX_PER_CM > 0 else 0.0

                score = float(confs[i]) if confs is not None else 1.0

                cx = (head[0] + tail[0]) / 2.0
                cy = (head[1] + tail[1]) / 2.0

                det = Detection(
                    points=np.array([[cx, cy]]),
                    scores=np.array([score]),
                    data={
                        "head": head,
                        "tail": tail,
                        "length_cm": float(length_cm),
                    },
                )
                detections.append(det)

        tracks = tracker.update(detections=detections)

        # Gambar pada frame
        for track in tracks:
            if track.last_detection is None:
                continue
            data = track.last_detection.data

            head = data["head"]
            tail = data["tail"]
            length_cm = data["length_cm"]
            track_id = track.id

            p1 = (int(head[0]), int(head[1]))
            p2 = (int(tail[0]), int(tail[1]))

            cv2.line(frame, p1, p2, (0, 255, 0), 2)

            text = f"ID {track_id} | {length_cm:.1f} cm"
            cx = int((head[0] + tail[0]) / 2)
            cy = int((head[1] + tail[1]) / 2) - 10
            cy = max(cy, 20)

            cv2.putText(
                frame,
                text,
                (cx, cy),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 0),
                2,
                cv2.LINE_AA,
            )

            records.append(
                {
                    "run_id": run_id,
                    "frame_index": frame_idx,
                    "track_id": int(track_id),
                    "length_cm": float(length_cm),
                }
            )

        total_detections += len(tracks)
        out.write(frame)

    cap.release()
    out.release()

    t1 = time.time()
    elapsed = t1 - t0

    if len(records) > 0:
        df = pd.DataFrame(records)
        df.to_csv(output_csv_path, index=False)
        mean_length = df["length_cm"].mean()
        max_length = df["length_cm"].max()
        min_length = df["length_cm"].min()
        unique_ids = df["track_id"].nunique()
    else:
        df = pd.DataFrame(columns=["run_id", "frame_index", "track_id", "length_cm"])
        df.to_csv(output_csv_path, index=False)
        mean_length = max_length = min_length = 0.0
        unique_ids = 0

    summary = {
        "run_id": run_id,
        "frames": frame_idx,
        "total_detections": int(total_detections),
        "unique_fish_ids": int(unique_ids),
        "mean_length_cm": float(mean_length),
        "max_length_cm": float(max_length),
        "min_length_cm": float(min_length),
        "processing_time_sec": float(elapsed),
    }

    return output_video_name, output_csv_name, summary


# =================================================================
# ROUTES FLASK
# =================================================================

@app.route("/", methods=["GET"])
def index():
    return render_template("index.html")


@app.route("/upload", methods=["POST"])
def upload_video():
    if "video" not in request.files:
        return jsonify({"status": "error", "message": "File video tidak ditemukan."}), 400

    file = request.files["video"]
    if file.filename == "":
        return jsonify({"status": "error", "message": "Nama file kosong."}), 400

    save_path = os.path.join(UPLOAD_DIR, file.filename)
    file.save(save_path)

    try:
        video_name, csv_name, summary = process_video(save_path)
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500

    return jsonify(
        {
            "status": "ok",
            "video_url": url_for("get_output_file", filename=video_name),
            "csv_url": url_for("get_output_file", filename=csv_name),
            "summary": summary,
        }
    )


@app.route("/outputs/<path:filename>")
def get_output_file(filename):
    return send_from_directory(OUTPUT_DIR, filename, as_attachment=False)


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000, debug=True)
