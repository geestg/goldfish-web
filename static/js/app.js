const form = document.getElementById("upload-form");
const imageInput = document.getElementById("image-input");
const statusBox = document.getElementById("status");

const previewOriginal = document.getElementById("preview-original");
const previewAnnotated = document.getElementById("preview-annotated");

const summaryBox = document.getElementById("summary-box");
const fishTableBody = document.querySelector("#fish-table tbody");
const csvLink = document.getElementById("csv-link");
const submitBtn = document.getElementById("btn-submit");

function showStatus(text, type = "info") {
  statusBox.textContent = text;
  statusBox.classList.remove("hidden", "info", "error");
  statusBox.classList.add(type);
}

if (form) {
  form.addEventListener("submit", async (e) => {
    e.preventDefault();

    if (!imageInput.files || imageInput.files.length === 0) {
      showStatus("Silakan pilih satu gambar terlebih dahulu.", "error");
      return;
    }

    const formData = new FormData();
    formData.append("image", imageInput.files[0]);

    // tampilkan pratinjau gambar asli
    const fileUrl = URL.createObjectURL(imageInput.files[0]);
    previewOriginal.src = fileUrl;

    // reset hasil sebelumnya
    previewAnnotated.src = "";
    fishTableBody.innerHTML = "";
    summaryBox.innerHTML = "";
    csvLink.href = "#";

    showStatus("Gambar sedang diproses. Mohon tunggu…", "info");
    submitBtn.disabled = true;
    submitBtn.textContent = "Memproses…";

    try {
      const resp = await fetch("/api/analyze-image", {
        method: "POST",
        body: formData,
      });

      if (!resp.ok) {
        let msg = "Terjadi kesalahan pada server.";
        try {
          const err = await resp.json();
          if (err && err.message) msg = err.message;
        } catch (_) {}
        showStatus(msg, "error");
        return;
      }

      const data = await resp.json();
      if (data.status !== "ok") {
        showStatus(data.message || "Proses gagal.", "error");
        return;
      }

      showStatus("Analisis selesai.", "info");

      // tampilkan gambar anotasi
      previewAnnotated.src = data.image_url;

      // ringkasan
      const s = data.summary;
      summaryBox.innerHTML = `
        <p><strong>Run ID:</strong> ${s.run_id}</p>
        <p><strong>Jumlah ikan terdeteksi:</strong> ${s.num_fish}</p>
        <p><strong>Panjang maksimum:</strong> ${s.max_length_cm.toFixed(2)} cm</p>
        <p><strong>Panjang minimum:</strong> ${s.min_length_cm.toFixed(2)} cm</p>
      `;

      // detail per ikan
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

      // link CSV
      csvLink.href = data.csv_url;
    } catch (err) {
      console.error(err);
      showStatus("Terjadi error saat menghubungi server.", "error");
    } finally {
      submitBtn.disabled = false;
      submitBtn.textContent = "Analisis Gambar";
    }
  });
}
