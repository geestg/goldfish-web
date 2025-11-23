// static/js/app.js
document.addEventListener("DOMContentLoaded", function () {
  const form = document.getElementById("uploadForm");
  const overlay = document.getElementById("loadingOverlay");

  if (form && overlay) {
    form.addEventListener("submit", function () {
      overlay.classList.add("active");
    });
  }
});
