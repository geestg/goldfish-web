// ======= Element Referensi =======
const form = document.getElementById("upload-form");
const statusBox = document.getElementById("status");
const resultSection = document.getElementById("result-section");
const resultVideo = document.getElementById("result-video");
const csvLink = document.getElementById("csv-link");
const summaryList = document.getElementById("summary-list");
const submitBtn = document.getElementById("btn-submit");

// ======= Helper Status =======
function showStatus(message, type = "info") {
  statusBox.textContent = message;
  statusBox.classList.remove("hidden", "info", "error");
  statusBox.classList.add(type);
}

// ======= Form Submit =======
if (form) {
  form.addEventListener("submit", async (e) => {
    e.preventDefault();

    const fileInput = document.getElementById("video");
    if (!fileInput.files || fileInput.files.length === 0) {
      showStatus("Silakan pilih berkas video terlebih dahulu.", "error");
      return;
    }

    const formData = new FormData();
    formData.append("video", fileInput.files[0]);

    // Lock UI saat proses
    if (submitBtn) {
      submitBtn.disabled = true;
      submitBtn.textContent = "Sedang menganalisis...";
    }
    showStatus("Video sedang diunggah dan dianalisis. Mohon tunggu.", "info");
    resultSection.classList.add("hidden");

    try {
      const response = await fetch("/upload", {
        method: "POST",
        body: formData,
      });

      if (!response.ok) {
        let msg = "Terjadi kesalahan pada server.";
        try {
          const err = await response.json();
          if (err && err.message) msg = err.message;
        } catch (_) {}
        showStatus(msg, "error");
        return;
      }

      const data = await response.json();
      if (data.status !== "ok") {
        showStatus(data.message || "Proses gagal.", "error");
        return;
      }

      // Sukses
      showStatus("Analisis selesai. Artefak hasil tersedia pada bagian Hasil Analisis.", "info");
      resultSection.classList.remove("hidden");

      // Update video & CSV
      resultVideo.src = data.video_url;
      csvLink.href = data.csv_url;

      // Ringkasan
      summaryList.innerHTML = "";
      if (data.summary) {
        const s = data.summary;
        const items = [
          ["Run ID", s.run_id],
          ["Jumlah frame", s.frames],
          ["Total deteksi (akumulatif)", s.total_detections],
          ["Jumlah ID ikan unik", s.unique_fish_ids],
          ["Rata-rata panjang ikan (cm)", s.mean_length_cm?.toFixed(2)],
          ["Panjang maksimum (cm)", s.max_length_cm?.toFixed(2)],
          ["Panjang minimum (cm)", s.min_length_cm?.toFixed(2)],
          ["Waktu pemrosesan (detik)", s.processing_time_sec?.toFixed(2)],
        ];

        items.forEach(([label, value]) => {
          const li = document.createElement("li");
          li.textContent = `${label}: ${value}`;
          summaryList.appendChild(li);
        });
      }

      // Setelah selesai, otomatis scroll ke section hasil
      const resultSectionWrapper = document.querySelector("#section-result");
      if (resultSectionWrapper) {
        resultSectionWrapper.scrollIntoView({ behavior: "smooth", block: "start" });
      }
    } catch (err) {
      console.error(err);
      showStatus("Terjadi error saat menghubungi server.", "error");
    } finally {
      if (submitBtn) {
        submitBtn.disabled = false;
        submitBtn.textContent = "Jalankan Analisis";
      }
    }
  });
}

// ======= Navigasi Sidebar (scroll ke section) =======
const navItems = document.querySelectorAll(".nav-item");

navItems.forEach((btn) => {
  btn.addEventListener("click", () => {
    const target = btn.getAttribute("data-target");
    const section = target ? document.querySelector(target) : null;
    if (section) {
      section.scrollIntoView({ behavior: "smooth", block: "start" });
    }

    navItems.forEach((b) => b.classList.remove("active"));
    btn.classList.add("active");
  });
});
