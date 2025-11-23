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
from norfair import Detection, Tracker


# ============================================================
# KONFIGURASI DASAR
# ============================================================

BASE_DIR = Path(__file__).resolve().parent

UPLOAD_DIR = BASE_DIR / "uploads"
OUTPUT_DIR = BASE_DIR / "outputs"
MODEL_PATH = BASE_DIR / "models" / "best.pt"

UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Faktor kalibrasi — bisa Anda update sesuai eksperimen
PX_PER_CM = float(os.environ.get("PX_PER_CM", 25.0))

# Skip frame agar pemrosesan stabil
FRAME_STEP = int(os.environ.get("FRAME_STEP", 3))

# Batas kategori ikan
BIG_FISH_THRESHOLD = float(os.environ.get("BIG_FISH_THRESHOLD", 12.0))
SMALL_FISH_MIN = float(os.environ.get("SMALL_FISH_MIN", 6.0))

# Load YOLOv8-Pose
model = YOLO(str(MODEL_PATH))

app = Flask(__name__)
app.secret_key = "goldfish-secret-key"


# ============================================================
# FUNGSI BANTU
# ============================================================

def hitung_panjang_ikan_kepala_ekor(keypoints_xy, px_per_cm: float):
    """
    Mengukur jarak kepala-ekor berdasarkan keypoints YOLO-Pose.
    Index 0=head, 1=tail.
    Noise filter: 4–40 cm.
    """
    panjang_cm = []
    for inst in keypoints_xy:
        head = inst[0]
        tail = inst[1]
        dist_px = np.linalg.norm(head - tail)
        dist_cm = dist_px / px_per_cm

        # Filter noise
        if 4.0 <= dist_cm <= 40.0:
            panjang_cm.append(dist_cm)

    return panjang_cm


def distance_norfair(detection: Detection, tracked_object):
    """
    Metode jarak Norfair: rata-rata Euclidean distance dari dua titik (head, tail).
    """
    return np.linalg.norm(detection.points - tracked_object.estimate, axis=1).mean()


# NORFAIR Tracker
tracker = Tracker(
    distance_function=distance_norfair,
    distance_threshold=35,   # sensitifitas tracking
)


# ============================================================
# FUNGSI UTAMA PEMROSESAN VIDEO
# ============================================================

def proses_video(video_path: Path, px_per_cm: float = PX_PER_CM):

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Tidak dapat membuka video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 25
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    run_id = uuid.uuid4().hex[:8]
    out_video = f"annotated_{run_id}.mp4"
    out_csv = f"summary_{run_id}.csv"

    out_video_path = OUTPUT_DIR / out_video
    out_csv_path = OUTPUT_DIR / out_csv

    # Encoder H.264 agar bisa diputar di browser
    fourcc = cv2.VideoWriter_fourcc(*"avc1")
    writer = cv2.VideoWriter(str(out_video_path), fourcc, fps, (width, height))

    # Reset tracker setiap video baru
    global tracker
    tracker = Tracker(distance_function=distance_norfair, distance_threshold=35)

    # Penyimpanan data panjang per ID
    length_history = {}
    all_lengths = []

    frame_index = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_index += 1

        # Skip sebagian frame
        if FRAME_STEP > 1 and (frame_index % FRAME_STEP) != 0:
            writer.write(frame)
            continue

        # Deteksi YOLOv8-Pose
        results = model(frame, verbose=False)[0]

        detections = []

        if results.keypoints is not None:
            kpts_xy = results.keypoints.xy.cpu().numpy()

            # Hitung panjang
            lengths = hitung_panjang_ikan_kepala_ekor(kpts_xy, px_per_cm)

            # Sinkronisasi
            n = min(len(kpts_xy), len(lengths))

            for i in range(n):
                head = kpts_xy[i][0]
                tail = kpts_xy[i][1]
                length_cm = float(lengths[i])

                points = np.vstack([head, tail])
                det = Detection(points=points, data={"length_cm": length_cm})
                detections.append(det)

        # Update tracker
        tracked = tracker.update(detections)

        # Anotasi
        for obj in tracked:
            tid = obj.id
            head_est, tail_est = obj.estimate

            # Ambil panjang dari deteksi terakhir
            if obj.last_detection is None:
                continue
            length_cm = obj.last_detection.data["length_cm"]

            # Simpan histori
            all_lengths.append(length_cm)
            length_history.setdefault(tid, []).append(length_cm)

            p1 = (int(head_est[0]), int(head_est[1]))
            p2 = (int(tail_est[0]), int(tail_est[1]))

            cv2.line(frame, p1, p2, (0, 170, 255), 2)
            cv2.circle(frame, p1, 5, (0, 102, 204), -1)
            cv2.circle(frame, p2, 5, (0, 204, 102), -1)

            label = f"ID {tid} | {length_cm:.1f} cm"
            cv2.putText(frame, label, (p1[0], p1[1] - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
            cv2.putText(frame, label, (p1[0], p1[1] - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

        writer.write(frame)

    cap.release()
    writer.release()

    # ============================================================
    # RINGKASAN HASIL
    # ============================================================

    if not all_lengths or not length_history:
        return {
            "run_id": run_id,
            "input_video_name": video_path.name,
            "output_video_name": out_video,
            "output_csv_name": out_csv,
            "summary": {
                "estimated_fish_count": 0,
                "fish_big_cm": None,
                "fish_small_cm": None,
                "min_length_cm": None,
                "max_length_cm": None,
                "fish_details": [],
            }
        }

    all_arr = np.array(all_lengths)
    min_len = float(all_arr.min())
    max_len = float(all_arr.max())

    fish_details = []
    for tid, vals in length_history.items():
        if vals:
            fish_details.append({
                "id": f"Ikan {tid}",
                "mean_length_cm": float(np.mean(vals)),
            })

    # Urutkan berdasarkan ID
    fish_details.sort(key=lambda x: int(x["id"].split()[-1]))

    # Kategori besar/kecil
    big_vals = [f["mean_length_cm"] for f in fish_details if f["mean_length_cm"] >= BIG_FISH_THRESHOLD]
    small_vals = [f["mean_length_cm"] for f in fish_details if SMALL_FISH_MIN <= f["mean_length_cm"] < BIG_FISH_THRESHOLD]

    summary = {
        "estimated_fish_count": len(fish_details),
        "fish_big_cm": float(np.mean(big_vals)) if big_vals else None,
        "fish_small_cm": float(np.mean(small_vals)) if small_vals else None,
        "min_length_cm": min_len,
        "max_length_cm": max_len,
        "fish_details": fish_details,
    }

    # Simpan CSV
    with open(out_csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["run_id", run_id])
        w.writerow(["video_name", video_path.name])
        w.writerow(["px_per_cm", px_per_cm])
        w.writerow([])
        w.writerow(["metric", "value"])
        for k, v in summary.items():
            w.writerow([k, v])

    return {
        "run_id": run_id,
        "input_video_name": video_path.name,
        "output_video_name": out_video,
        "output_csv_name": out_csv,
        "summary": summary,
    }


# ============================================================
# ROUTES
# ============================================================

def list_previous_runs():
    return sorted([x.name for x in OUTPUT_DIR.glob("summary_*.csv")])


@app.route("/", methods=["GET"])
def index():
    return render_template("index.html", result=None, runs=list_previous_runs())


@app.route("/upload", methods=["POST"])
def upload_and_process():
    if "video" not in request.files:
        flash("Tidak ada berkas video yang dikirim.")
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
        result = proses_video(save_path, px_per_cm=PX_PER_CM)
    except Exception as e:
        flash(f"Terjadi kesalahan: {e}")
        return redirect(url_for("index"))

    return render_template("index.html", result=result, runs=list_previous_runs())


@app.route("/outputs/video/<path:filename>")
def get_output_video(filename):
    return send_from_directory(OUTPUT_DIR, filename)


@app.route("/outputs/csv/<path:filename>")
def get_output_csv(filename):
    return send_from_directory(OUTPUT_DIR, filename, as_attachment=True)


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000, debug=True)
