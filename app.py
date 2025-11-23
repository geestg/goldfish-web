# app.py
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

# ------------------ Konfigurasi dasar ------------------

BASE_DIR = Path(__file__).resolve().parent

UPLOAD_DIR = Path(os.environ.get("UPLOAD_DIR", BASE_DIR / "uploads"))
OUTPUT_DIR = Path(os.environ.get("OUTPUT_DIR", BASE_DIR / "outputs"))
MODEL_PATH = Path(os.environ.get("MODEL_PATH", BASE_DIR / "models" / "best.pt"))

UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Faktor kalibrasi (px per cm). Silakan sesuaikan dengan hasil kalibrasi Anda.
PX_PER_CM = float(os.environ.get("PX_PER_CM", 25.0))

# Load model YOLOv8-Pose
model = YOLO(str(MODEL_PATH))

app = Flask(__name__)
app.secret_key = "goldfish-secret-key"  # diperlukan untuk flash message


# ------------------ Fungsi pengukuran ------------------

def hitung_panjang_ikan_kepala_ekor(keypoints_xy, px_per_cm: float):
    """
    keypoints_xy : array (N, K, 2) -> N instance, K keypoints, (x, y)
    Asumsi: indeks 0 = head, indeks 1 = tail.
    Mengembalikan list panjang ikan dalam piksel dan cm.
    """
    panjang_px, panjang_cm = [], []
    for inst in keypoints_xy:
        head = inst[0]
        tail = inst[1]
        dist_px = float(np.linalg.norm(head - tail))
        dist_cm = dist_px / px_per_cm
        panjang_px.append(dist_px)
        panjang_cm.append(dist_cm)
    return panjang_px, panjang_cm


def proses_video(video_path: Path, px_per_cm: float = PX_PER_CM):
    """
    Memproses satu video:
      - Jalankan YOLOv8-Pose per frame
      - Hitung panjang ikan
      - Simpan video beranotasi ke OUTPUT_DIR
      - Simpan ringkasan ke file CSV
    """
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Tidak dapat membuka video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    run_id = uuid.uuid4().hex[:8]
    out_video_name = f"annotated_{run_id}.mp4"
    out_video_path = OUTPUT_DIR / out_video_name
    out_csv_name = f"summary_{run_id}.csv"
    out_csv_path = OUTPUT_DIR / out_csv_name

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(out_video_path), fourcc, fps, (width, height))

    all_lengths_cm = []
    fish_counts = []
    frame_index = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_index += 1

        results = model(frame, verbose=False)[0]

        if results.keypoints is not None:
            kpts_xy = results.keypoints.xy.cpu().numpy()  # (N, K, 2)
            lengths_px, lengths_cm = hitung_panjang_ikan_kepala_ekor(
                kpts_xy, px_per_cm
            )

            all_lengths_cm.extend(lengths_cm)
            fish_counts.append(len(lengths_cm))

            # Gambar anotasi ke frame
            for i, inst in enumerate(kpts_xy):
                head = inst[0]
                tail = inst[1]
                length_cm = lengths_cm[i]

                p1 = (int(head[0]), int(head[1]))
                p2 = (int(tail[0]), int(tail[1]))

                cv2.line(frame, p1, p2, (0, 170, 255), 2)
                cv2.circle(frame, p1, 5, (0, 102, 204), -1)
                cv2.circle(frame, p2, 5, (0, 204, 102), -1)

                label = f"{length_cm:.1f} cm"
                cv2.putText(
                    frame,
                    label,
                    (p1[0], max(p1[1] - 10, 20)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (0, 0, 0),
                    3,
                    cv2.LINE_AA,
                )
                cv2.putText(
                    frame,
                    label,
                    (p1[0], max(p1[1] - 10, 20)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (255, 255, 255),
                    1,
                    cv2.LINE_AA,
                )

        writer.write(frame)

    cap.release()
    writer.release()

    if not all_lengths_cm:
        summary = {
            "estimated_fish_count": 0,
            "fish_A_cm": None,
            "fish_B_cm": None,
            "mean_length_cm": None,
            "min_length_cm": None,
            "max_length_cm": None,
        }
    else:
        mean_len = float(np.mean(all_lengths_cm))
        min_len = float(np.min(all_lengths_cm))
        max_len = float(np.max(all_lengths_cm))

        try:
            est_count = int(statistics.mode(fish_counts))
        except statistics.StatisticsError:
            est_count = int(round(np.mean(fish_counts)))

        summary = {
            "estimated_fish_count": est_count,
            "fish_A_cm": max_len,
            "fish_B_cm": min_len,
            "mean_length_cm": mean_len,
            "min_length_cm": min_len,
            "max_length_cm": max_len,
        }

    # Simpan ringkasan ke CSV
    with open(out_csv_path, "w", newline="", encoding="utf-8") as f:
        writer_csv = csv.writer(f)
        writer_csv.writerow(["run_id", run_id])
        writer_csv.writerow(["video_name", video_path.name])
        writer_csv.writerow(["px_per_cm", px_per_cm])
        writer_csv.writerow([])
        writer_csv.writerow(["metric", "value"])
        for key, value in summary.items():
            writer_csv.writerow([key, value])

    return {
        "run_id": run_id,
        "input_video_name": video_path.name,
        "output_video_name": out_video_name,
        "output_csv_name": out_csv_name,
        "summary": summary,
    }


# ------------------ Route Flask ------------------

def list_previous_runs():
    summaries = []
    for csv_file in sorted(OUTPUT_DIR.glob("summary_*.csv")):
        summaries.append(csv_file.name)
    return summaries


@app.route("/", methods=["GET"])
def index():
    runs = list_previous_runs()
    return render_template("index.html", result=None, runs=runs)


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
