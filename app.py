import os
import re
import uuid
from datetime import datetime

import cv2
import numpy as np
import pandas as pd
from flask import (
    Flask,
    render_template,
    request,
    jsonify,
    Response,
    send_file,
)
from ultralytics import YOLO


# ============================================================
# KONFIGURASI
# ============================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

UPLOAD_DIR = os.path.join(BASE_DIR, "uploads")
WEB_OUTPUT_IMAGE = os.path.join(BASE_DIR, "analisa_gambar")
WEB_OUTPUT_VIDEO = os.path.join(BASE_DIR, "analisa_video")
MODEL_PATH = os.path.join(BASE_DIR, "models", "best.pt")

os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(WEB_OUTPUT_IMAGE, exist_ok=True)
os.makedirs(WEB_OUTPUT_VIDEO, exist_ok=True)

PX_PER_CM = 25.0  # nilai kalibrasi pixel → cm


# ============================================================
# FLASK INIT
# ============================================================

app = Flask(__name__, static_folder="static", template_folder="templates")
print(f"[INFO] Model Loaded: {MODEL_PATH}")
model = YOLO(MODEL_PATH)


def run_id():
    return datetime.now().strftime("%Y%m%d-%H%M%S") + "-" + uuid.uuid4().hex[:5]


# ============================================================
# DRAWING ANNOTATIONS
# ============================================================

def draw_annotations(img, box, head, tail, length_cm):
    x1, y1, x2, y2 = map(int, box)

    # kotak kuning
    cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 255), 2)

    # titik head merah
    cv2.circle(img, (int(head[0]), int(head[1])), 6, (0, 0, 255), -1)

    # titik tail hijau
    cv2.circle(img, (int(tail[0]), int(tail[1])), 6, (0, 255, 0), -1)

    # garis antar head-tail
    cv2.line(img, (int(head[0]), int(head[1])), (int(tail[0]), int(tail[1])),
             (0, 255, 0), 3)

    # label panjang
    label = f"{length_cm:.2f} cm"
    cv2.putText(img, label, (x1, y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                (0, 255, 255), 2)


# ============================================================
# ANALISIS GAMBAR
# ============================================================

def analyze_image(img_path):
    rid = run_id()
    img = cv2.imread(img_path)
    res = model(img)[0]

    annotated = img.copy()
    records = []

    if res.keypoints is not None:
        kpts = res.keypoints.xy.cpu().numpy()
        boxes = res.boxes.xyxy.cpu().numpy()
        confs = res.boxes.conf.cpu().numpy()

        for i in range(len(kpts)):
            head = kpts[i, 0]
            tail = kpts[i, 1]

            px = float(np.linalg.norm(head - tail))
            cm = px / PX_PER_CM

            draw_annotations(annotated, boxes[i], head, tail, cm)

            records.append({
                "run_id": rid,
                "fish_id": i + 1,
                "confidence": float(confs[i]),
                "length_px": px,
                "length_cm": cm
            })

    idx = len(os.listdir(WEB_OUTPUT_IMAGE)) + 1
    img_name = f"IMG_ANALYSIS_{idx:04d}.png"
    csv_name = f"IMG_ANALYSIS_{idx:04d}.csv"

    cv2.imwrite(os.path.join(WEB_OUTPUT_IMAGE, img_name), annotated)
    pd.DataFrame(records).to_csv(os.path.join(WEB_OUTPUT_IMAGE, csv_name), index=False)

    summary = {
        "run_id": rid,
        "num_fish": len(records),
        "max_length_cm": max([r["length_cm"] for r in records], default=0),
        "min_length_cm": min([r["length_cm"] for r in records], default=0),
    }

    return img_name, csv_name, summary, records


# ============================================================
# ANALISIS VIDEO
# ============================================================

def analyze_video(video_path):
    rid = run_id()

    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 15
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    idx = len(os.listdir(WEB_OUTPUT_VIDEO)) + 1
    out_video = f"VID_ANALYSIS_{idx:04d}.mp4"
    out_csv = f"VID_ANALYSIS_{idx:04d}.csv"

    out_vpath = os.path.join(WEB_OUTPUT_VIDEO, out_video)
    csv_path = os.path.join(WEB_OUTPUT_VIDEO, out_csv)

    # gunakan H.264
    fourcc = cv2.VideoWriter_fourcc(*"avc1")
    writer = cv2.VideoWriter(out_vpath, fourcc, fps, (w, h))

    logs = []
    frame_idx = 0

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        res = model(frame)[0]
        annotated = frame.copy()

        if res.keypoints is not None:
            kpts = res.keypoints.xy.cpu().numpy()
            boxes = res.boxes.xyxy.cpu().numpy()

            for i in range(len(kpts)):
                head = kpts[i, 0]
                tail = kpts[i, 1]

                px = float(np.linalg.norm(head - tail))
                cm = px / PX_PER_CM

                draw_annotations(annotated, boxes[i], head, tail, cm)

                logs.append({
                    "frame": frame_idx,
                    "fish_id": i + 1,
                    "length_cm": cm
                })

        writer.write(annotated)
        frame_idx += 1

    cap.release()
    writer.release()

    pd.DataFrame(logs).to_csv(csv_path, index=False)

    return out_video, out_csv, rid, len(logs), logs


# ============================================================
# ROUTE FILE ANOTASI GAMBAR (FIX UTAMA)
# ============================================================

@app.route("/analisa_gambar/<path:filename>")
def serve_analysis_image(filename):
    return send_file(os.path.join(WEB_OUTPUT_IMAGE, filename))


@app.route("/analisa_gambar/csv/<path:filename>")
def serve_analysis_image_csv(filename):
    return send_file(os.path.join(WEB_OUTPUT_IMAGE, filename))


@app.route("/analisa_video/csv/<path:filename>")
def serve_analysis_video_csv(filename):
    return send_file(os.path.join(WEB_OUTPUT_VIDEO, filename))


# ============================================================
# VIDEO STREAMING
# ============================================================

@app.route("/analisa_video/<path:filename>")
def stream_video(filename):
    file_path = os.path.join(WEB_OUTPUT_VIDEO, filename)

    if not os.path.exists(file_path):
        return "Not Found", 404

    file_size = os.path.getsize(file_path)
    range_header = request.headers.get("Range")

    if range_header:
        match = re.search(r"bytes=(\d+)-(\d*)", range_header)
        start = int(match.group(1))
        end = match.group(2)
        end = int(end) if end else file_size - 1

        chunk = end - start + 1

        with open(file_path, "rb") as f:
            f.seek(start)
            data = f.read(chunk)

        resp = Response(data, 206, mimetype="video/mp4")
        resp.headers["Content-Range"] = f"bytes {start}-{end}/{file_size}"
        resp.headers["Accept-Ranges"] = "bytes"
        resp.headers["Content-Length"] = str(chunk)
        return resp

    return send_file(file_path, mimetype="video/mp4")


# ============================================================
# ROUTE HALAMAN
# ============================================================

@app.route("/")
def index():
    return render_template("index.html", active="home")


@app.route("/image")
def page_image():
    return render_template("image.html", active="image")


@app.route("/video")
def page_video():
    return render_template("video.html", active="video")


# ============================================================
# API
# ============================================================

@app.route("/api/analyze-image", methods=["POST"])
def api_image():
    f = request.files["image"]
    saved = os.path.join(UPLOAD_DIR, f.filename)
    f.save(saved)

    img_name, csv_name, summary, records = analyze_image(saved)

    return jsonify({
        "status": "ok",
        "summary": summary,
        "records": records,
        "image_url": f"/analisa_gambar/{img_name}",
        "csv_url": f"/analisa_gambar/{csv_name}",
    })


@app.route("/api/analyze-video", methods=["POST"])
def api_video():
    f = request.files["video"]
    saved = os.path.join(UPLOAD_DIR, f.filename)
    f.save(saved)

    video_name, csv_name, rid, total_logs, logs = analyze_video(saved)

    return jsonify({
        "status": "ok",
        "run_id": rid,
        "video_url": f"/analisa_video/{video_name}",
        "csv_url": f"/analisa_video/{csv_name}",
        "total_logs": total_logs,
        "records": logs
    })


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=8000)
