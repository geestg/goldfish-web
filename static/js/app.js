// =====================
// Helper status
// =====================
function setStatus(el, text, type = "info") {
  if (!el) return;
  el.textContent = text;
  el.classList.remove("hidden", "info", "error");
  el.classList.add(type);
}

// =====================
// Analisis gambar
// =====================
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

    if (!imageInput.files || imageInput.files.length === 0) {
      setStatus(imageStatus, "Silakan pilih gambar terlebih dahulu.", "error");
      return;
    }

    const formData = new FormData();
    formData.append("image", imageInput.files[0]);

    // pratinjau gambar
    if (previewOriginal) {
      previewOriginal.src = URL.createObjectURL(imageInput.files[0]);
    }
    if (previewAnnotated) previewAnnotated.src = "";
    if (summaryBox) summaryBox.innerHTML = "";
    if (fishTableBody) fishTableBody.innerHTML = "";
    if (csvLink) csvLink.href = "#";

    setStatus(imageStatus, "Mengirim gambar ke server…", "info");
    if (btnImage) btnImage.disabled = true;

    try {
      const resp = await fetch("/api/analyze-image", {
        method: "POST",
        body: formData,
      });

      const data = await resp.json();

      if (!resp.ok || data.status !== "ok") {
        setStatus(
          imageStatus,
          data.message || "Terjadi kesalahan saat analisis.",
          "error"
        );
        return;
      }

      setStatus(imageStatus, "Analisis selesai.", "info");

      if (previewAnnotated) {
        previewAnnotated.src = data.image_url;
      }

      const s = data.summary;
      if (summaryBox) {
        summaryBox.classList.remove("muted");
        summaryBox.innerHTML = `
          <p><strong>Run ID:</strong> ${s.run_id}</p>
          <p><strong>Jumlah ikan:</strong> ${s.num_fish}</p>
          <p><strong>Panjang maksimum:</strong> ${s.max_length_cm.toFixed(2)} cm</p>
          <p><strong>Panjang minimum:</strong> ${s.min_length_cm.toFixed(2)} cm</p>
        `;
      }

      if (fishTableBody) {
        fishTableBody.innerHTML = "";
        (data.records || []).forEach((r) => {
          const tr = document.createElement("tr");
          tr.innerHTML = `
            <td>${r.fish_id}</td>
            <td>${r.confidence.toFixed(3)}</td>
            <td>${r.length_cm.toFixed(2)}</td>
          `;
          fishTableBody.appendChild(tr);
        });
      }

      if (csvLink) {
        csvLink.href = data.csv_url;
      }
    } catch (err) {
      console.error(err);
      setStatus(imageStatus, "Gagal menghubungi server.", "error");
    } finally {
      if (btnImage) btnImage.disabled = false;
    }
  });
}

// =====================
// Analisis video
// =====================
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

    if (!videoInput.files || videoInput.files.length === 0) {
      setStatus(videoStatus, "Silakan pilih file video terlebih dahulu.", "error");
      return;
    }

    const formData = new FormData();
    formData.append("video", videoInput.files[0]);

    setStatus(videoStatus, "Video sedang diproses…", "info");
    if (btnVideo) btnVideo.disabled = true;

    try {
      const resp = await fetch("/api/analyze-video", {
        method: "POST",
        body: formData,
      });

      const data = await resp.json();
      if (!resp.ok || data.status !== "ok") {
        setStatus(
          videoStatus,
          data.message || "Terjadi kesalahan saat analisis video.",
          "error"
        );
        return;
      }

      setStatus(videoStatus, "Analisis video selesai.", "info");

      if (videoPreview) {
        videoPreview.src = data.video_url;
      }
      if (videoCsv) {
        videoCsv.href = data.csv_url;
      }
      if (videoSummary) {
        videoSummary.classList.remove("muted");
        videoSummary.innerHTML = `
          <p><strong>Run ID:</strong> ${data.run_id}</p>
          <p><strong>Total log deteksi:</strong> ${data.total_logs}</p>
        `;
      }
    } catch (err) {
      console.error(err);
      setStatus(videoStatus, "Gagal menghubungi server.", "error");
    } finally {
      if (btnVideo) btnVideo.disabled = false;
    }
  });
}
