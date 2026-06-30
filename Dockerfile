# 1. Pake base image Python yang stabil dan slim (ramping)
FROM python:3.10-slim

# 2. Set folder kerja di dalam kontainer Docker
WORKDIR /code

# 3. Install dependency OS yang dibutuhin OpenCV buat processing gambar
RUN apt-get update && apt-get install -y \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    && rm -rf /var/lib/apt/lists/*

# 4. Copy file requirements dulu (biar Docker caching-nya optimal)
COPY ./requirements.txt /code/requirements.txt

# 5. Install semua library Python tanpa bikin cache sampah
RUN pip install --no-cache-dir --upgrade -r /code/requirements.txt

# 6. Copy seluruh source code aplikasi kita ke dalam Docker
COPY ./app /code/app

# 7. Buka port 8000 (port bawaan FastAPI kita)
EXPOSE 8000

# 8. Perintah untuk menyalakan FastAPI pake Uvicorn pas kontainer jalan
# Kita matikan --reload di production VPS agar performanya maksimal dan stabil
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]