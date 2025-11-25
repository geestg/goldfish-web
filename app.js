const PUBLISHABLE_KEY = "rf_20YOFKNHDhPshQr6ay0wPxEKwh72";
const MODEL_NAME = "goldfish-yolo-pose-5j7bv";
const MODEL_VERSION = 9;
const PX_PER_CM = 25;

let model = null;

const fileInput = document.getElementById("fileInput");
const imgInput = document.getElementById("imgInput");
const videoInput = document.getElementById("videoInput");

const canvas = document.getElementById("canvasOutput");
const ctx = canvas.getContext("2d");

const statusBox = document.getElementById("status");
const logBox = document.getElementById("logBox");
const summaryBox = document.getElementById("summaryBox");

function log(t) {
    logBox.textContent += t + "\n";
}

async function loadModel() {
    try {
        model = await roboflow
            .auth({ publishable_key: PUBLISHABLE_KEY })
            .load({ model: MODEL_NAME, version: MODEL_VERSION });

        statusBox.textContent = "Model siap!";
        statusBox.className = "status ready";
        log("Model berhasil dimuat.");
    } catch (err) {
        statusBox.textContent = "Gagal memuat model.";
        statusBox.className = "status error";
        log("ERROR: " + err);
    }
}

loadModel();

fileInput.addEventListener("change", event => {
    const file = event.target.files[0];
    if (!file) return;

    const isVideo = file.type.startsWith("video/");
    const isImage = file.type.startsWith("image/");

    imgInput.style.display = "none";
    videoInput.style.display = "none";

    if (isImage) {
        imgInput.src = URL.createObjectURL(file);
        imgInput.onload = () => runImage(imgInput);
        imgInput.style.display = "block";
    }

    if (isVideo) {
        videoInput.src = URL.createObjectURL(file);
        videoInput.onloadeddata = () => runVideo(videoInput);
        videoInput.style.display = "block";
    }
});

async function runImage(img) {
    canvas.width = img.width;
    canvas.height = img.height;

    ctx.drawImage(img, 0, 0);
    const preds = await model.predict(img);

    drawPredictions(preds);
    showSummary(preds);
}

async function runVideo(video) {
    const fps = 10;
    const interval = 1000 / fps;

    async function processFrame() {
        if (video.paused || video.ended) return;

        canvas.width = video.videoWidth;
        canvas.height = video.videoHeight;

        ctx.drawImage(video, 0, 0);

        const preds = await model.predict(canvas);
        drawPredictions(preds);
        showSummary(preds);

        setTimeout(processFrame, interval);
    }

    processFrame();
}

function drawPredictions(preds) {
    preds.forEach(pred => {
        const x = pred.bbox.x - pred.bbox.width / 2;
        const y = pred.bbox.y - pred.bbox.height / 2;

        ctx.strokeStyle = "yellow";
        ctx.lineWidth = 3;
        ctx.strokeRect(x, y, pred.bbox.width, pred.bbox.height);

        if (pred.keypoints && pred.keypoints.length >= 2) {
            const head = pred.keypoints[0];
            const tail = pred.keypoints[1];

            const dx = tail.x - head.x;
            const dy = tail.y - head.y;

            const length_px = Math.sqrt(dx*dx + dy*dy);
            const length_cm = (length_px / PX_PER_CM).toFixed(2);

            ctx.fillStyle = "red";
            ctx.beginPath();
            ctx.arc(head.x, head.y, 4, 0, 2*Math.PI);
            ctx.fill();

            ctx.beginPath();
            ctx.arc(tail.x, tail.y, 4, 0, 2*Math.PI);
            ctx.fill();

            ctx.fillStyle = "yellow";
            ctx.font = "18px Inter";
            ctx.fillText(`${length_cm} cm`, x + 5, y - 10);

            pred.length_cm = length_cm;
        }

        log(JSON.stringify(pred, null, 2));
    });
}

function showSummary(preds) {
    const count = preds.length;
    const lengths = preds
        .filter(p => p.length_cm)
        .map(p => parseFloat(p.length_cm));

    let avg = 0;
    if (lengths.length > 0)
        avg = (lengths.reduce((a,b)=>a+b,0) / lengths.length).toFixed(2);

    summaryBox.innerHTML = `
        <p><strong>Jumlah ikan terdeteksi:</strong> ${count}</p>
        <p><strong>Panjang rata-rata:</strong> ${avg} cm</p>
    `;
}
