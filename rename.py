import os

# Path folder gambar
folder_path = r"D:\goldfish-web\datasets"

# Ambil semua file gambar
files = os.listdir(folder_path)
image_files = [f for f in files if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
image_files.sort()

# Tahap 1: rename sementara (tambah prefix tmp_)
for idx, filename in enumerate(image_files):
    ext = os.path.splitext(filename)[1]
    old_path = os.path.join(folder_path, filename)
    tmp_path = os.path.join(folder_path, f"tmp_{idx}{ext}")
    os.rename(old_path, tmp_path)

# Tahap 2: ambil ulang semua file tmp_ dan rename final
tmp_files = [f for f in os.listdir(folder_path) if f.startswith("tmp_")]
tmp_files.sort()

for idx, filename in enumerate(tmp_files, start=1):
    ext = os.path.splitext(filename)[1]
    old_path = os.path.join(folder_path, filename)
    new_path = os.path.join(folder_path, f"mas_{idx}{ext}")
    os.rename(old_path, new_path)
    print(f"Renamed: {filename} → mas_{idx}{ext}")

print("✅ Semua gambar berhasil di-rename tanpa konflik!")
