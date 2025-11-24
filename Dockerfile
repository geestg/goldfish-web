FROM python:3.10

# OpenCV dependencies
RUN apt-get update && apt-get install -y \
    libgl1 \
    libglib2.0-0 \
    ffmpeg && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .

RUN pip install --upgrade pip wheel setuptools

# Numpy versi lama (2.0 ke atas bikin error)
RUN pip install "numpy<2.0"

# Install torch CPU
RUN pip install torch==2.1.0 torchvision==0.16.0 --index-url https://download.pytorch.org/whl/cpu

# 💡 ULTRALYTICS KOMPATIBEL DENGAN MODEL ANDA
RUN pip install ultralytics==8.0.20

# OpenCV
RUN pip install opencv-python-headless==4.8.1.78

# Tracking
RUN pip install norfair==2.2.0

COPY . .

EXPOSE 8000

CMD ["gunicorn", "--bind", "0.0.0.0:8000", "app:app"]
