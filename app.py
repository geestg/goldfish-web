# app.py
import os
import uuid
import statistics
from pathlib import Path

import cv2
import numpy as np
from flask import Flask, render_template, request, redirect, url_for, jsonify
from ultralytics import YOLO

# ================== KONFIGURASI DASAR ==================

# Direktori video (akan kita mount dari Google Drive)
VIDEO_DIR = Path(os.environ.get("VIDEO_DIR", "/videos"))  # nanti di Docker pakai -v
OUTPUT_DIR = Path(os.environ.get("OUTPUT_DIR", "/outputs"))
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Faktor kalibrasi sementara (px per cm) – nanti bisa disesuaikan hasil kalibrasi
PX_PER_CM = float(os.environ.get("PX_PER_CM", 25.0))

# Lokasi model
MODEL_PATH = os.environ.get("MODEL_PATH", "models/best.pt")

# Load model YOLOv8 Pose sekali di awal
model = YOLO(MODEL_PATH)

app = Flask(__name__)


# ================== FUNGSI PENGUKURAN PANJANG ==================

def hitung_panjang_ikan_kepala_ekor(keypoints_xy, px_per_cm=PX_PER_CM):
    """
    keypoints_xy: array (num_instances, num_kpts, 2)
    Mengembalikan list panjang ikan dalam piksel dan cm
    """
    panjang_px = []
    panjang_cm = []

    # asumsi: keypoint 0 = head, 1 = tail
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
    - Hitung estimasi jumlah ikan
    - Simpan video hasil anotasi
    """
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Tidak dapat membuka video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Nama file output
    out_name = f"processed_{uuid.uuid4().hex}.mp4"
    out_path = OUTPUT_DIR / out_name

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(out_path), fourcc, fps, (width, height))

    all_lengths_cm = []
    fish_counts = []
    frame_idx = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_idx += 1

        # Inference YOLOv8 Pose
        results = model(frame, verbose=False)[0]

        if results.keypoints is not None:
            kpts_xy = results.keypoints.xy.cpu().numpy()  # (N, K, 2)
            lengths_px, lengths_cm = hitung_panjang_ikan_kepala_ekor(
                kpts_xy, px_per_cm
            )

            fish_counts.append(len(lengths_cm))
            all_lengths_cm.extend(lengths_cm)

            # Anotasi ke frame
            for i, inst in enumerate(kpts_xy):
                head = inst[0]
                tail = inst[1]
                length_cm = lengths_cm[i]

                p1 = (int(head[0]), int(head[1]))
                p2 = (int(tail[0]), int(tail[1]))

                cv2.line(frame, p1, p2, (0, 255, 0), 2)
                cv2.circle(frame, p1, 4, (0, 0, 255), -1)
                cv2.circle(frame, p2, 4, (255, 0, 0), -1)
                label = f"{length_cm:.1f} cm"
                cv2.putText(
                    frame,
                    label,
                    (p1[0], p1[1] - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (0, 255, 0),
                    2,
                    cv2.LINE_AA,
                )

        writer.write(frame)

    cap.release()
    writer.release()

    if not all_lengths_cm:
        # Tidak ada ikan terdeteksi
        summary = {
            "estimated_fish_count": 0,
            "fish_A_cm": None,
            "fish_B_cm": None,
            "mean_length_cm": None,
            "min_length_cm": None,
            "max_length_cm": None,
        }
    else:
        # Ringkasan statistik
        mean_len = float(np.mean(all_lengths_cm))
        min_len = float(np.min(all_lengths_cm))
        max_len = float(np.max(all_lengths_cm))

        # Estimasi jumlah ikan: modus jumlah deteksi per frame
        try:
            est_count = int(statistics.mode(fish_counts))
        except statistics.StatisticsError:
            est_count = int(round(np.mean(fish_counts)))

        # Ikan A: paling besar, Ikan B: paling kecil
        fish_A_cm = max_len
        fish_B_cm = min_len

        summary = {
            "estimated_fish_count": est_count,
            "fish_A_cm": fish_A_cm,
            "fish_B_cm": fish_B_cm,
            "mean_length_cm": mean_len,
            "min_length_cm": min_len,
            "max_length_cm": max_len,
        }

    return {
        "output_video_name": out_name,
        "output_video_path": str(out_path),
        "summary": summary,
    }


# ================== ROUTES FLASK ==================

def list_videos():
    VIDEO_DIR.mkdir(parents=True, exist_ok=True)
    return sorted(
        [f.name for f in VIDEO_DIR.iterdir() if f.is_file()],
        key=lambda x: x.lower()
    )


@app.route("/", methods=["GET"])
def index():
    videos = list_videos()
    return render_template(
        "index.html",
        videos=videos,
    )


@app.route("/process", methods=["POST"])
def process():
    """
    Endpoint 'Cek Ikan':
    - Bisa memilih video dari daftar Google Drive (mount ke /videos)
    - Bisa upload video baru (opsional)
    """
    selected_video = request.form.get("selected_video")
    uploaded = request.files.get("uploaded_video")

    if uploaded and uploaded.filename:
        # Simpan video upload ke VIDEO_DIR
        VIDEO_DIR.mkdir(parents=True, exist_ok=True)
        up_path = VIDEO_DIR / uploaded.filename
        uploaded.save(str(up_path))
        video_path = up_path
        source_type = "upload"
    elif selected_video:
        video_path = VIDEO_DIR / selected_video
        source_type = "drive"
    else:
        return redirect(url_for("index"))

    result = proses_video(video_path)

    return render_template(
        "index.html",
        videos=list_videos(),
        result=result,
        selected_video=video_path.name,
        source_type=source_type,
    )


@app.route("/api/process", methods=["POST"])
def api_process():
    """
    API JSON (bisa dipakai nanti untuk integrasi dashboard):
    body: { "video_name": "xxx.mp4" }
    """
    data = request.get_json(force=True)
    video_name = data.get("video_name")
    video_path = VIDEO_DIR / video_name

    result = proses_video(video_path)
    return jsonify(result)


if __name__ == "__main__":
    # Untuk pengujian lokal (non-Docker)
    app.run(host="0.0.0.0", port=8000, debug=True)
