# Dockerfile
FROM python:3.10-slim

WORKDIR /app

# Install dependency dasar
RUN apt-get update && apt-get install -y \
    libgl1 libglib2.0-0 && \
    rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Lokasi default model dan direktori IO
ENV MODEL_PATH=/app/models/best.pt
ENV VIDEO_DIR=/videos
ENV OUTPUT_DIR=/outputs
ENV PX_PER_CM=25.0

# Buat folder output
RUN mkdir -p /outputs

EXPOSE 8000

CMD ["gunicorn", "-b", "0.0.0.0:8000", "app:app"]
