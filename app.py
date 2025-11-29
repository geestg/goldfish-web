import os
from datetime import datetime
import uuid

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


# ===================================================================
# PATH
# ===================================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

UPLOAD_DIR = os.path.join(BASE_DIR, "uploads")

WEB_OUTPUT_IMAGE = r"D:\goldfish-web\analisa_gambar"
WEB_OUTPUT_VIDEO = r"D:\goldfish-web\analisa_video"

MODEL_PATH = os.path.join(BASE_DIR, "models", "best.pt")

os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(WEB_OUTPUT_IMAGE, exist_ok=True)
os.makedirs(WEB_OUTPUT_VIDEO, exist_ok=True)

# konversi pixel → cm
PX_PER_CM = 25.0

app = Flask(__name__, static_folder="static", template_folder="templates")

print(f"[INFO] Loading YOLO Pose Model: {MODEL_PATH}")
model = YOLO(MODEL_PATH)


def generate_run_id():
    return datetime.now().strftime("%Y%m%d-%H%M%S") + "-" + uuid.uuid4().hex[:5]


# ===================================================================
# ANALISIS GAMBAR
# ===================================================================

def analyze_image(image_path: str):
    run_id = generate_run_id()

    img = cv2.imread(image_path)
    if img is None:
        raise RuntimeError("Gambar tidak dapat dibaca.")

    results = model(img, verbose=False)
    res = results[0]

    annotated = img.copy()
    records = []

    # proses keypoint
    if res.keypoints is not None:
        kpts = res.keypoints.xy.cpu().numpy()
        confs = res.boxes.conf.cpu().numpy() if res.boxes else None

        for i in range(kpts.shape[0]):
            head = kpts[i, 0]
            tail = kpts[i, 1]

            length_px = float(np.linalg.norm(head - tail))
            length_cm = length_px / PX_PER_CM
            score = float(confs[i]) if confs is not None else 1.0

            # gambar garis & titik
            cv2.line(
                annotated, (int(head[0]), int(head[1])), (int(tail[0]), int(tail[1])),
                (0, 255, 0), 2
            )
            cv2.circle(annotated, (int(head[0]), int(head[1])), 4, (0, 0, 255), -1)
            cv2.circle(annotated, (int(tail[0]), int(tail[1])), 4, (0, 0, 255), -1)

            # label panjang
            cv2.putText(
                annotated,
                f"{length_cm:.1f} cm",
                (int(head[0]), int(head[1]) - 8),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (0, 255, 0),
                2,
            )

            records.append({
                "run_id": run_id,
                "fish_id": i + 1,
                "confidence": score,
                "length_px": length_px,
                "length_cm": length_cm,
            })

    # ==============================================================
    # SIMPAN BERURUTAN
    # ==============================================================

    output_idx = len(os.listdir(WEB_OUTPUT_IMAGE)) + 1
    annotated_name = f"IMG_ANALYSIS_{output_idx:04d}.png"
    csv_name = f"IMG_ANALYSIS_{output_idx:04d}.csv"

    annotated_path = os.path.join(WEB_OUTPUT_IMAGE, annotated_name)
    csv_path = os.path.join(WEB_OUTPUT_IMAGE, csv_name)

    cv2.imwrite(annotated_path, annotated)
    pd.DataFrame(records).to_csv(csv_path, index=False)

    summary = {
        "run_id": run_id,
        "num_fish": len(records),
        "max_length_cm": max([r["length_cm"] for r in records], default=0),
        "min_length_cm": min([r["length_cm"] for r in records], default=0),
    }

    return annotated_name, csv_name, summary, records


# ===================================================================
# ANALISIS VIDEO (FIX 100% – TIDAK HANG)
# ===================================================================

def analyze_video(video_path: str):
    run_id = generate_run_id()

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError("Video gagal dibuka oleh OpenCV.")

    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps == 0 or fps is None:
        fps = 15  # default aman

    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # nama output berurutan
    output_idx = len(os.listdir(WEB_OUTPUT_VIDEO)) + 1
    out_video_name = f"VID_ANALYSIS_{output_idx:04d}.mp4"
    csv_name = f"VID_ANALYSIS_{output_idx:04d}.csv"

    out_video_path = os.path.join(WEB_OUTPUT_VIDEO, out_video_name)
    csv_path = os.path.join(WEB_OUTPUT_VIDEO, csv_name)

    # FIX codec Windows
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer = cv2.VideoWriter(out_video_path, fourcc, float(fps), (w, h))

    logs = []
    frame_idx = 0

    while True:
        ret, frame = cap.read()

        # =============================
        # FIX: berhenti jika frame kosong
        # =============================
        if not ret or frame is None:
            break

        try:
            results = model(frame, verbose=False)
        except Exception as e:
            print("[ERROR] YOLO gagal memproses frame:", e)
            continue

        res = results[0]
        annotated = frame.copy()

        # proses keypoint
        if res.keypoints is not None:
            kpts = res.keypoints.xy.cpu().numpy()

            for i in range(kpts.shape[0]):
                head = kpts[i, 0]
                tail = kpts[i, 1]

                length_px = float(np.linalg.norm(head - tail))
                length_cm = length_px / PX_PER_CM

                # gambar
                cv2.line(
                    annotated,
                    (int(head[0]), int(head[1])),
                    (int(tail[0]), int(tail[1])),
                    (0, 255, 0),
                    2,
                )

                logs.append({
                    "frame": frame_idx,
                    "fish_id": i + 1,
                    "length_cm": length_cm,
                })

        writer.write(annotated)
        frame_idx += 1

        # =============================
        # FIX: proteksi infinite loop
        # =============================
        if frame_idx > total_frames + 5:
            print("[WARN] Loop frame melebihi batas.")
            break

    cap.release()
    writer.release()

    pd.DataFrame(logs).to_csv(csv_path, index=False)

    return out_video_name, csv_name, run_id, len(logs)


# ===================================================================
# ROUTE PAGES
# ===================================================================

@app.route("/")
def index():
    return render_template("index.html", title="Dashboard", active="home")


@app.route("/image")
def page_image():
    return render_template("image.html", title="Analisis Gambar", active="image")


@app.route("/video")
def page_video():
    return render_template("video.html", title="Analisis Video", active="video")


# ===================================================================
# API ANALISIS GAMBAR
# ===================================================================

@app.route("/api/analyze-image", methods=["POST"])
def api_analyze_image():
    if "image" not in request.files:
        return jsonify({"status": "error", "message": "Tidak ada file gambar."}), 400

    file = request.files["image"]
    save_path = os.path.join(UPLOAD_DIR, file.filename)
    file.save(save_path)

    try:
        annotated_name, csv_name, summary, records = analyze_image(save_path)
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500

    return jsonify({
        "status": "ok",
        "summary": summary,
        "records": records,
        "image_url": f"/analisa_gambar/{annotated_name}",
        "csv_url": f"/analisa_gambar/{csv_name}",
    })


# ===================================================================
# API ANALISIS VIDEO
# ===================================================================

@app.route("/api/analyze-video", methods=["POST"])
def api_analyze_video():
    if "video" not in request.files:
        return jsonify({"status": "error", "message": "Tidak ada file video."}), 400

    file = request.files["video"]
    save_path = os.path.join(UPLOAD_DIR, file.filename)
    file.save(save_path)

    try:
        video_name, csv_name, run_id, total_logs = analyze_video(save_path)
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500

    return jsonify({
        "status": "ok",
        "run_id": run_id,
        "video_url": f"/analisa_video/{video_name}",
        "csv_url": f"/analisa_video/{csv_name}",
        "total_logs": total_logs,
    })


# ===================================================================
# SERVE OUTPUT FILES
# ===================================================================

@app.route("/analisa_gambar/<path:filename>")
def serve_img_output(filename):
    return send_from_directory(WEB_OUTPUT_IMAGE, filename)


@app.route("/analisa_video/<path:filename>")
def serve_vid_output(filename):
    return send_from_directory(WEB_OUTPUT_VIDEO, filename)


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000, debug=True)
