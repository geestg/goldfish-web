function setStatus(el, text, type = "info") {
  if (!el) return;
  el.textContent = text;
  el.classList.remove("hidden", "error", "info");
  el.classList.add(type);
}

/* ============================================================
   ANALISIS GAMBAR
============================================================ */

const imageForm = document.getElementById("image-form");
const imageInput = document.getElementById("image-input");
const imageStatus = document.getElementById("image-status");
const btnImage = document.getElementById("btn-image");

const previewOriginal = document.getElementById("preview-original");
const previewAnnotated = document.getElementById("preview-annotated");
const summaryBox = document.getElementById("summary-box");
const fishTableBody = document.querySelector("#fish-table tbody");
const csvLink = document.getElementById("csv-link");

if (imageForm) {
  imageForm.addEventListener("submit", async (e) => {
    e.preventDefault();

    const form = new FormData();
    form.append("image", imageInput.files[0]);

    previewOriginal.src = URL.createObjectURL(imageInput.files[0]);
    previewAnnotated.src = "";

    setStatus(imageStatus, "Memproses gambar...", "info");
    btnImage.disabled = true;

    const resp = await fetch("/api/analyze-image", {
      method: "POST",
      body: form,
    });

    const data = await resp.json();
    btnImage.disabled = false;

    if (data.status !== "ok") {
      setStatus(imageStatus, data.message, "error");
      return;
    }

    previewAnnotated.src = data.image_url + "?v=" + Date.now();
    csvLink.href = data.csv_url;

    summaryBox.innerHTML = `
      <p><strong>Run ID:</strong> ${data.summary.run_id}</p>
      <p><strong>Jumlah ikan:</strong> ${data.summary.num_fish}</p>
      <p><strong>Panjang Maksimum:</strong> ${data.summary.max_length_cm.toFixed(2)} cm</p>
      <p><strong>Panjang Minimum:</strong> ${data.summary.min_length_cm.toFixed(2)} cm</p>
    `;
    summaryBox.classList.remove("muted");

    fishTableBody.innerHTML = "";
    data.records.forEach((r) => {
      const row = `
        <tr>
          <td>${r.fish_id}</td>
          <td>${r.confidence.toFixed(3)}</td>
          <td>${r.length_cm.toFixed(2)}</td>
        </tr>`;
      fishTableBody.insertAdjacentHTML("beforeend", row);
    });

    setStatus(imageStatus, "Analisis selesai.", "info");
  });
}

/* ============================================================
   ANALISIS VIDEO
============================================================ */

const videoForm = document.getElementById("video-form");
const videoInput = document.getElementById("video-input");
const videoStatus = document.getElementById("video-status");
const btnVideo = document.getElementById("btn-video");

const videoPreview = document.getElementById("video-preview");
const videoCsv = document.getElementById("video-csv");
const videoSummary = document.getElementById("video-summary");

if (videoForm) {
  videoForm.addEventListener("submit", async (e) => {
    e.preventDefault();

    const form = new FormData();
    form.append("video", videoInput.files[0]);

    setStatus(videoStatus, "Memproses video...", "info");
    btnVideo.disabled = true;

    const resp = await fetch("/api/analyze-video", {
      method: "POST",
      body: form,
    });

    const data = await resp.json();
    btnVideo.disabled = false;

    if (data.status !== "ok") {
      setStatus(videoStatus, data.message, "error");
      return;
    }

    videoPreview.src = data.video_url + "?v=" + Date.now();
    videoPreview.load();
    videoPreview.play().catch(() => {});

    videoCsv.href = data.csv_url;

    videoSummary.innerHTML = `
      <p><strong>Run ID:</strong> ${data.run_id}</p>
      <p><strong>Total log deteksi:</strong> ${data.total_logs}</p>
    `;
    videoSummary.classList.remove("muted");

    setStatus(videoStatus, "Analisis Video selesai.", "info");
  });
}

/* ============================================================
   STREAMING LARIX + YOLO (REALTIME)
============================================================ */

const streamImg = document.getElementById("stream-video");
const streamStatus = document.getElementById("stream-status");
const btnStreamCapture = document.getElementById("btn-stream-capture");
const btnStreamRecStart = document.getElementById("btn-stream-rec-start");
const btnStreamRecStop = document.getElementById("btn-stream-rec-stop");

if (streamImg) {

  // Capture snapshot
  btnStreamCapture.addEventListener("click", async () => {
    setStatus(streamStatus, "Menyimpan snapshot...", "info");

    const resp = await fetch("/stream/capture", { method: "POST" });
    const data = await resp.json();

    if (data.status !== "ok") {
      setStatus(streamStatus, data.message || "Gagal capture.", "error");
      return;
    }

    setStatus(streamStatus, "Snapshot disimpan: " + data.file, "info");
  });

  // Mulai rekam
  btnStreamRecStart.addEventListener("click", async () => {
    setStatus(streamStatus, "Mengaktifkan perekaman...", "info");

    const resp = await fetch("/stream/record-start", { method: "POST" });
    const data = await resp.json();

    if (data.status === "error") {
      setStatus(streamStatus, data.message, "error");
    } else if (data.status === "already_recording") {
      setStatus(streamStatus, "Sedang merekam.", "info");
    } else {
      setStatus(streamStatus, "Rekaman dimulai: " + data.file, "info");
    }
  });

  // Stop rekam
  btnStreamRecStop.addEventListener("click", async () => {
    const resp = await fetch("/stream/record-stop", { method: "POST" });
    const data = await resp.json();

    if (data.status === "ok") {
      setStatus(streamStatus, "Rekaman dihentikan.", "info");
    } else {
      setStatus(streamStatus, "Tidak ada rekaman aktif.", "error");
    }
  });

}
