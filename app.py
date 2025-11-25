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

# =========================
# Konfigurasi dasar
# =========================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

UPLOAD_DIR = os.path.join(BASE_DIR, "uploads")
OUTPUT_DIR = os.path.join(BASE_DIR, "outputs")
MODEL_PATH = os.path.join(BASE_DIR, "models", "best.pt")

os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)

PX_PER_CM = 25.0  # kalibrasi pixel -> cm

# =========================
# Load model YOLO Pose
# =========================
print(f"[INFO] Loading Roboflow Pose Model: {MODEL_PATH}")
model = YOLO(MODEL_PATH)

app = Flask(__name__)


def generate_run_id():
    return datetime.now().strftime("%Y%m%d-%H%M%S") + "-" + uuid.uuid4().hex[:6]


def analyze_image(image_path: str):
    run_id = generate_run_id()

    img = cv2.imread(image_path)
    if img is None:
        raise RuntimeError("Gambar tidak dapat dibaca.")

    h, w = img.shape[:2]

    # =============== INFERENSI POSE ===============
    results = model(img, verbose=False)
    res = results[0]

    annotated = img.copy()
    records = []

    if res.keypoints is not None:
        kpts = res.keypoints.xy.cpu().numpy()  # [N, keypoints, 2]
        num_instances = kpts.shape[0]

        boxes = res.boxes.xyxy.cpu().numpy() if res.boxes is not None else None
        confs = res.boxes.conf.cpu().numpy() if res.boxes is not None else None

        for i in range(num_instances):
            head = kpts[i, 0]
            tail = kpts[i, 1]

            length_px = float(np.linalg.norm(head - tail))
            length_cm = length_px / PX_PER_CM

            score = float(confs[i]) if confs is not None else 1.0

            if boxes is not None:
                x1, y1, x2, y2 = boxes[i]
            else:
                xs = [head[0], tail[0]]
                ys = [head[1], tail[1]]
                x1, y1, x2, y2 = min(xs), min(ys), max(xs), max(ys)

            p_head = (int(head[0]), int(head[1]))
            p_tail = (int(tail[0]), int(tail[1]))

            cv2.rectangle(annotated, (int(x1), int(y1)), (int(x2), int(y2)),
                          (0, 255, 255), 2)
            cv2.circle(annotated, p_head, 4, (0, 0, 255), -1)
            cv2.circle(annotated, p_tail, 4, (0, 0, 255), -1)
            cv2.line(annotated, p_head, p_tail, (0, 255, 0), 2)

            label = f"ID {i+1} | {length_cm:.1f} cm"
            cv2.putText(annotated, label, (int(x1), int(y1)-10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                        (0, 255, 0), 2)

            records.append({
                "run_id": run_id,
                "fish_id": i+1,
                "confidence": score,
                "length_px": length_px,
                "length_cm": length_cm,
                "x1": float(x1),
                "y1": float(y1),
                "x2": float(x2),
                "y2": float(y2),
            })

    annotated_name = f"{run_id}_annotated.png"
    csv_name = f"{run_id}_summary.csv"
    annotated_path = os.path.join(OUTPUT_DIR, annotated_name)
    csv_path = os.path.join(OUTPUT_DIR, csv_name)

    cv2.imwrite(annotated_path, annotated)

    if records:
        df = pd.DataFrame(records)
        df.to_csv(csv_path, index=False)

        lengths = [r["length_cm"] for r in records]
        summary = {
            "run_id": run_id,
            "num_fish": len(records),
            "mean_length_cm": float(np.mean(lengths)),
            "max_length_cm": float(np.max(lengths)),
            "min_length_cm": float(np.min(lengths)),
        }
    else:
        summary = {
            "run_id": run_id,
            "num_fish": 0,
            "mean_length_cm": 0.0,
            "max_length_cm": 0.0,
            "min_length_cm": 0.0,
        }

    return annotated_name, csv_name, summary, records, (h, w)


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/api/analyze-image", methods=["POST"])
def api_analyze_image():
    if "image" not in request.files:
        return jsonify({"status": "error", "message": "File gambar tidak ditemukan."}), 400

    file = request.files["image"]
    if file.filename == "":
        return jsonify({"status": "error", "message": "Nama file kosong."}), 400

    save_path = os.path.join(UPLOAD_DIR, file.filename)
    file.save(save_path)

    try:
        annotated_name, csv_name, summary, records, shape = analyze_image(save_path)
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500

    return jsonify({
        "status": "ok",
        "summary": summary,
        "records": records,
        "image_url": url_for("get_output_file", filename=annotated_name),
        "csv_url": url_for("get_output_file", filename=csv_name),
        "image_shape": {"height": shape[0], "width": shape[1]},
    })


@app.route("/outputs/<path:filename>")
def get_output_file(filename):
    return send_from_directory(OUTPUT_DIR, filename)


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000, debug=True)
