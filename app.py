import os
import uuid
import statistics
from pathlib import Path
import csv

import cv2
import numpy as np
from flask import (
    Flask,
    render_template,
    request,
    redirect,
    url_for,
    send_from_directory,
    flash,
)
from ultralytics import YOLO


# ---------------------------------------------------
# Konfigurasi dasar
# ---------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent
UPLOAD_DIR = BASE_DIR / "uploads"
OUTPUT_DIR = BASE_DIR / "outputs"
MODEL_PATH = BASE_DIR / "models" / "best.pt"

UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

PX_PER_CM = float(os.environ.get("PX_PER_CM", 25.0))
FRAME_STEP = int(os.environ.get("FRAME_STEP", 4))

BIG_FISH_THRESHOLD = float(os.environ.get("BIG_FISH_THRESHOLD", 12.0))
SMALL_FISH_MIN = float(os.environ.get("SMALL_FISH_MIN", 6.0))

# Load model YOLO
model = YOLO(str(MODEL_PATH))

app = Flask(__name__)
app.secret_key = "goldfish-secret-key"


# ---------------------------------------------------
# Fungsi bantu
# ---------------------------------------------------

def hitung_panjang_ikan_kepala_ekor(keypoints_xy, px_per_cm: float):
    panjang_px, panjang_cm = [], []
    for inst in keypoints_xy:
        head = inst[0]
        tail = inst[1]
        dist_px = float(np.linalg.norm(head - tail))
        dist_cm = dist_px / px_per_cm

        if 3.0 <= dist_cm <= 30.0:
            panjang_px.append(dist_px)
            panjang_cm.append(dist_cm)

    return panjang_px, panjang_cm


def iou_xyxy(box1, box2):
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])

    inter_w = max(0.0, x2 - x1)
    inter_h = max(0.0, y2 - y1)
    inter_area = inter_w * inter_h
    if inter_area <= 0:
        return 0.0

    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union = area1 + area2 - inter_area
    if union <= 0:
        return 0.0

    return float(inter_area / union)


class SimpleFishTracker:
    def __init__(self, iou_threshold=0.35, max_lost=15):
        self.next_id = 1
        self.tracks = {}
        self.iou_threshold = iou_threshold
        self.max_lost = max_lost

    def update(self, detections, frame_idx):
        assigned_tracks = set()
        results = []

        for tid in list(self.tracks.keys()):
            if self.tracks[tid]["last_frame"] < frame_idx:
                self.tracks[tid]["lost"] += 1
                if self.tracks[tid]["lost"] > self.max_lost:
                    del self.tracks[tid]

        for det in detections:
            db = det["bbox"]
            best_iou = 0.0
            best_id = None

            for tid, tinfo in self.tracks.items():
                if tid in assigned_tracks:
                    continue
                iou = iou_xyxy(tinfo["bbox"], db)
                if iou > best_iou:
                    best_iou = iou
                    best_id = tid

            if best_id is not None and best_iou >= self.iou_threshold:
                self.tracks[best_id]["bbox"] = db
                self.tracks[best_id]["last_frame"] = frame_idx
                self.tracks[best_id]["lost"] = 0
                assigned_tracks.add(best_id)

                results.append(
                    {"track_id": best_id, "bbox": db, "length_cm": det["length_cm"]}
                )
            else:
                tid = self.next_id
                self.next_id += 1

                self.tracks[tid] = {
                    "bbox": db,
                    "last_frame": frame_idx,
                    "lost": 0,
                }
                assigned_tracks.add(tid)

                results.append(
                    {"track_id": tid, "bbox": db, "length_cm": det["length_cm"]}
                )

        return results


# ---------------------------------------------------
# Fungsi utama pemrosesan video
# ---------------------------------------------------

def proses_video(video_path: Path, px_per_cm: float = PX_PER_CM):
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Tidak dapat membuka video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 25
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    run_id = uuid.uuid4().hex[:8]

    # ---- FIX: H.264 diganti AVI MJPG ----
    out_video_name = f"annotated_{run_id}.avi"
    out_video_path = OUTPUT_DIR / out_video_name
    fourcc = cv2.VideoWriter_fourcc(*"MJPG")

    writer = cv2.VideoWriter(str(out_video_path), fourcc, fps, (width, height))

    tracker = SimpleFishTracker()

    length_history = {}
    all_lengths_cm = []

    frame_index = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_index += 1

        if FRAME_STEP > 1 and frame_index % FRAME_STEP != 0:
            writer.write(frame)
            continue

        results = model(frame, verbose=False)[0]

        detections = []
        if results.keypoints is not None and results.boxes is not None:

            kpts_xy = results.keypoints.xy.cpu().numpy()
            boxes_xyxy = results.boxes.xyxy.cpu().numpy()

            _, lengths_cm = hitung_panjang_ikan_kepala_ekor(kpts_xy, px_per_cm)

            n = min(len(boxes_xyxy), len(lengths_cm))
            for i in range(n):
                detections.append(
                    {"bbox": boxes_xyxy[i].tolist(), "length_cm": float(lengths_cm[i])}
                )

        tracked = tracker.update(detections, frame_index)

        for obj in tracked:
            tid = obj["track_id"]
            bbox = obj["bbox"]
            length_cm = obj["length_cm"]

            all_lengths_cm.append(length_cm)
            length_history.setdefault(tid, []).append(length_cm)

            x1, y1, x2, y2 = map(int, bbox)

            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 200, 255), 2)
            label = f"ID {tid} | {length_cm:.1f} cm"

            cv2.putText(
                frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 3
            )
            cv2.putText(
                frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1
            )

        writer.write(frame)

    cap.release()
    writer.release()

    # ------------------ Ringkasan ------------------

    if not all_lengths_cm:
        summary = {
            "estimated_fish_count": 0,
            "fish_big_cm": None,
            "fish_small_cm": None,
            "min_length_cm": None,
            "max_length_cm": None,
            "fish_details": [],
        }
    else:
        all_arr = np.array(all_lengths_cm, dtype=float)
        min_len = float(all_arr.min())
        max_len = float(all_arr.max())

        fish_details = []
        for tid, vals in length_history.items():
            fish_details.append(
                {"id": f"Ikan {tid}", "mean_length_cm": float(np.mean(vals))}
            )

        fish_details.sort(key=lambda x: int(x["id"].split()[-1]))

        big_lengths = [
            d["mean_length_cm"] for d in fish_details if d["mean_length_cm"] >= BIG_FISH_THRESHOLD
        ]

        small_lengths = [
            d["mean_length_cm"]
            for d in fish_details
            if SMALL_FISH_MIN <= d["mean_length_cm"] < BIG_FISH_THRESHOLD
        ]

        summary = {
            "estimated_fish_count": len(fish_details),
            "fish_big_cm": float(np.mean(big_lengths)) if big_lengths else None,
            "fish_small_cm": float(np.mean(small_lengths)) if small_lengths else None,
            "min_length_cm": min_len,
            "max_length_cm": max_len,
            "fish_details": fish_details,
        }

    out_csv_name = f"summary_{run_id}.csv"
    out_csv_path = OUTPUT_DIR / out_csv_name

    with open(out_csv_path, "w", newline="", encoding="utf-8") as f:
        writer_csv = csv.writer(f)
        writer_csv.writerow(["run_id", run_id])
        writer_csv.writerow(["video_name", video_path.name])
        writer_csv.writerow(["px_per_cm", px_per_cm])
        writer_csv.writerow([])
        writer_csv.writerow(["metric", "value"])
        for k, v in summary.items():
            writer_csv.writerow([k, v])

    return {
        "run_id": run_id,
        "input_video_name": video_path.name,
        "output_video_name": out_video_name,
        "output_csv_name": out_csv_name,
        "summary": summary,
    }


# ---------------------------------------------------
# ROUTES FLASK
# ---------------------------------------------------

def list_previous_runs():
    return sorted([f.name for f in OUTPUT_DIR.glob("summary_*.csv")])


@app.route("/")
def index():
    runs = list_previous_runs()
    return render_template("index.html", result=None, runs=runs)


@app.route("/upload", methods=["POST"])
def upload_and_process():
    if "video" not in request.files:
        flash("Tidak ada berkas video.")
        return redirect(url_for("index"))

    file = request.files["video"]
    if file.filename == "":
        flash("Nama berkas kosong.")
        return redirect(url_for("index"))

    ext = Path(file.filename).suffix
    safe_name = f"input_{uuid.uuid4().hex[:8]}{ext}"

    save_path = UPLOAD_DIR / safe_name
    file.save(str(save_path))

    try:
        result = proses_video(save_path)
    except Exception as e:
        flash(f"Terjadi kesalahan saat memproses video: {e}")
        return redirect(url_for("index"))

    runs = list_previous_runs()
    return render_template("index.html", result=result, runs=runs)


@app.route("/outputs/video/<path:filename>")
def get_output_video(filename):
    return send_from_directory(OUTPUT_DIR, filename)


@app.route("/outputs/csv/<path:filename>")
def get_output_csv(filename):
    return send_from_directory(OUTPUT_DIR, filename, as_attachment=True)


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000, debug=True)
