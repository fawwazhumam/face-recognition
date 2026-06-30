import numpy as np
import cv2
import uuid
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from app.model_utils import face_pipeline
from app.database import get_db_connection

app = FastAPI(
    title="Face Recognition System",
    description="YOLOv12n-Face + GhostFaceNetV2 + pgvector",
    version="1.0"
)

def read_upload_file(file: UploadFile) -> np.ndarray:
    """convert to opencv format"""
    try:
        file_bytes = np.frombuffer(file.file.read(), np.uint8)
        img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
        if img is None:
            raise HTTPException(status_code=400, detail="image was broken")
        return img
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"failed to read image: {str(e)}")

@app.post("/register")
async def register_face(name: str = Form(...), file: UploadFile = File(...)):
    # 1. Cek ekstensi file yang dikirim client
    print(f"📁 DEBUG: Nama file yang diupload: {file.filename}")
    print(f"🧪 DEBUG: Tipe Content-Type: {file.content_type}")
    
    img = read_upload_file(file)
    
    # 2. Cek apakah OpenCV berhasil ngebaca filenya
    if img is None:
        print("❌ DEBUG CRITICAL: OpenCV GAGAL men-decode gambar! Matriks bernilai None.")
        raise HTTPException(status_code=400, detail="Format file rusak atau tidak didukung OpenCV!")
    else:
        print(f"📸 DEBUG SUCCESS: OpenCV sukses baca gambar! Ukuran matriks: {img.shape}")
        
    embedding = face_pipeline.detect_and_extract(img)
    
    if embedding is None:
        raise HTTPException(status_code=400, detail="face not detected!")
        
    # --- DI SINI TEMPAT KITA GENERATE ID ACAK AMAN ---
    # Membuat string unik acak, contoh hasil: '6fc9a321-7b3d-4958-bc42-4d56d34bde55'
    secure_user_id = str(uuid.uuid4())
    
    conn = get_db_connection()
    cur = conn.cursor()
    try:
        # PENTING: Struktur tabel users nanti kolom id-nya memakai tipe VARCHAR(50) atau UUID
        cur.execute(
            "INSERT INTO users (id, name) VALUES (%s, %s);", 
            (secure_user_id, name)
        )
        
        # Simpan relasi fitur wajah menggunakan id unik yang sudah di-secure tadi
        cur.execute(
            "INSERT INTO face_features (user_id, face_embedding) VALUES (%s, %s);",
            (secure_user_id, embedding)
        )
        conn.commit()
        return {"status": "success", "message": f"Face registered successfully with Secure ID: {secure_user_id}"}
    except Exception as e:
        conn.rollback()
        raise HTTPException(status_code=500, detail=f"Database error: {str(e)}")
    finally:
        cur.close()
        conn.close()

@app.post("/predict", summary="scan face for instant validation")
async def predict_face(file: UploadFile = File(...)):
    # 1. Konversi gambar upload ke OpenCV format
    img = read_upload_file(file)
    
    # 2. Ekstrak vektor wajah target menggunakan duet YOLOv12 + GhostFaceNet
    embedding = face_pipeline.detect_and_extract(img)
    
    if embedding is None:
        raise HTTPException(status_code=400, detail="face not detected by the scanner camera!")
        
    # 3. Lakukan pencarian kemiripan vektor menggunakan operator '<=>' (Cosine Distance) bawaan pgvector
    # Kita set threshold di angka 0.4 (Makin kecil nilainya, makin ketat/akurat pengenalannya)
    threshold = 0.5
    
    conn = get_db_connection()
    cur = conn.cursor()
    try:
        # Query elitis: cari 1 baris yang jarak cosine vektornya paling nempel (terkecil)
        cur.execute("""
            SELECT u.id, u.name, (f.face_embedding <=> %s::vector) AS distance
            FROM face_features f
            JOIN users u ON f.user_id = u.id
            WHERE (f.face_embedding <=> %s::vector) < %s
            ORDER BY distance ASC
            LIMIT 1;
        """, (embedding, embedding, threshold))
        
        result = cur.fetchone()
        
        if result:
            return {
                "status": "success",
                "match": True,
                "user_id": result['id'],
                "name": result['name'],
                "confidence_score": round(float(1 - result['distance']) * 100, 2) # Ubah distance ke persentase akurasi
            }
        else:
            return {
                "status": "success",
                "match": False,
                "message": "Face not recognized"
            }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to perform search: {str(e)}")
    finally:
        cur.close()
        conn.close()