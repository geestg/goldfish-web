const PUBLISHABLE_KEY = "rf_20YOFKNHDhPshQr6ay0wPxEKwh72";
const MODEL_NAME = "goldfish-yolo-pose-5j7bv";
const MODEL_VERSION = 9;

let model = null;

const uploadEl = document.getElementById("upload");
const imgEl = document.getElementById("input-image");
const canvasEl = document.getElementById("output");
const ctx = canvasEl.getContext("2d");

const statusEl = document.getElementById("status");
const logEl = document.getElementById("log");

function log(text) {
    logEl.textContent += text + "\n";
}

(async function loadModel() {
    try {
        model = await roboflow
            .auth({
                publishable_key: PUBLISHABLE_KEY
            })
            .load({
                model: MODEL_NAME,
                version: MODEL_VERSION
            });

        statusEl.textContent = "Model siap digunakan.";
        statusEl.className = "status ready";
        log("Model berhasil dimuat.");
    } catch (err) {
        statusEl.textContent = "Gagal memuat model!";
        statusEl.className = "status error";
        log("ERROR: " + err);
    }
})();

uploadEl.addEventListener("change", (event) => {
    const file = event.target.files[0];
    if (!file) return;

    imgEl.src = URL.createObjectURL(file);

    imgEl.onload = async () => {
        canvasEl.width = imgEl.naturalWidth;
        canvasEl.height = imgEl.naturalHeight;

        ctx.drawImage(imgEl, 0, 0);

        statusEl.textContent = "Memproses gambar...";
        statusEl.className = "status loading";

        try {
            const predictions = await model.predict(imgEl);

            log("Prediksi:");
            log(JSON.stringify(predictions, null, 2));

            ctx.drawImage(imgEl, 0, 0);

            predictions.forEach(pred => {
                const x = pred.bbox.x - pred.bbox.width / 2;
                const y = pred.bbox.y - pred.bbox.height / 2;

                ctx.strokeStyle = "yellow";
                ctx.lineWidth = 3;
                ctx.strokeRect(x, y, pred.bbox.width, pred.bbox.height);

                ctx.fillStyle = "yellow";
                ctx.font = "18px Inter";
                ctx.fillText(`${pred.class} (${Math.round(pred.confidence * 100)}%)`, x + 4, y + 20);

                if (pred.keypoints) {
                    ctx.fillStyle = "red";
                    pred.keypoints.forEach(kp => {
                        ctx.beginPath();
                        ctx.arc(kp.x, kp.y, 4, 0, 2 * Math.PI);
                        ctx.fill();
                    });
                }
            });

            statusEl.textContent = "Deteksi selesai.";
            statusEl.className = "status ready";

        } catch (err) {
            statusEl.textContent = "Terjadi error!";
            statusEl.className = "status error";
            log("ERROR: " + err);
        }
    };
});
