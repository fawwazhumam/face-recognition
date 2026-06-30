import os
import cv2
import numpy as np
import onnxruntime as ort

class FaceRecognitionPipeline:
    def __init__(self):
        # Jalur mutlak ke file model ONNX kita, bro
        self.yolo_path = "app/weights/yolov12n-face.onnx"
        self.ghost_path = "app/weights/ghostfacenetv2.onnx"
        
        # Pastikan file modelnya beneran ada sebelum dinyalakan sirkuitnya
        if not os.path.exists(self.yolo_path) or not os.path.exists(self.ghost_path):
            raise FileNotFoundError("Model weights tidak ditemukan di folder app/weights!")

        # Inisialisasi ONNX Runtime Session (Secara default pake CPU Execution Provider)
        self.yolo_session = ort.InferenceSession(self.yolo_path, providers=['CPUExecutionProvider'])
        self.ghost_session = ort.InferenceSession(self.ghost_path, providers=['CPUExecutionProvider'])
        
        # Ambil nama input dan output layer dari masing-masing model
        self.yolo_input_name = self.yolo_session.get_inputs()[0].name
        self.ghost_input_name = self.ghost_session.get_inputs()[0].name

    def _preprocess_yolo(self, img: np.ndarray, target_size=(640, 640)):
        """convert to yolo input format"""
        h, w, _ = img.shape
        # Ubah ukuran gambar ke 640x640
        resized = cv2.resize(img, target_size)
        
        # --- FIX BUG 1 ---
        # Karena di main.py sudah diubah ke RGB, di sini kita LANGSUNG normalisasi 
        # tanpa perlu cv2.cvtColor lagi agar warna asli wajah tidak rusak/terbalik!
        blob = resized.astype(np.float32) / 255.0
        blob = np.transpose(blob, (2, 0, 1))
        blob = np.expand_dims(blob, axis=0) # Tambah dimensi batch (1, 3, 640, 640)
        return blob, h, w

    def detect_and_extract(self, frame: np.ndarray):
        """
        Fungsi utama: Terima gambar frame kamera -> Deteksi Wajah -> Ekstrak Vektor 512
        """
        # --- TAHAP 1: DETEKSI WAJAH DENGAN YOLOV12N-FACE ---
        blob, orig_h, orig_w = self._preprocess_yolo(frame)
        
        # Jalankan inference YOLO
        yolo_outputs = self.yolo_session.run(None, {self.yolo_input_name: blob})
        predictions = yolo_outputs[0][0] # Ambil hasil batch pertama
        
        conf_threshold = 0.4  # Kita longgarkan sedikit ke 0.4 agar deteksi lokal lebih sensitif
        best_box = None
        max_conf = 0
        
        # --- FIX BUG 2 ---
        # Kita cek bentuk matriks output YOLO-nya. 
        # Jika barisnya lebih sedikit dari kolom (misal 16 x 8400), kita transpose (.T) 
        # agar susunannya menjadi (8400 x 16) sehingga bisa di-loop per objek prediksi.
        if predictions.shape[0] < predictions.shape[1]:
            predictions = predictions.T
            
        for pred in predictions:
            conf = pred[4] # Index ke-4 adalah skor keyakinan objek wajah
            if conf > conf_threshold and conf > max_conf:
                max_conf = conf
                best_box = pred[:4] # Ambil 4 koordinat kotak utamanya [x_center, y_center, w, h]
                
        if best_box is None:
            print("🤖 DEBUG MODEL: YOLOv12 gagal menemukan box wajah dengan confidence yang cukup.")
            return None

        # Kembalikan koordinat kotak dari skala 640 ke ukuran resolusi gambar asli lu
        x_center, y_center, w, h = best_box
        x1 = int((x_center - w / 2) * (orig_w / 640))
        y1 = int((y_center - h / 2) * (orig_h / 640))
        x2 = int((x_center + w / 2) * (orig_w / 640))
        y2 = int((y_center + h / 2) * (orig_h / 640))
        
        # Batasi koordinat agar tidak minus atau melebihi frame gambar (anti-crash)
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(orig_w, x2), min(orig_h, y2)
        
        # --- TAHAP 2: POTONG & PREPROCESS UNTUK GHOSTFACENETV2 ---
        face_crop = frame[y1:y2, x1:x2]
        if face_crop.size == 0:
            return None
            
        # GhostFaceNetV2 mewajibkan input wajah bersih berukuran 112x112 piksel
        face_resized = cv2.resize(face_crop, (112, 112))
        
        # Karena 'frame' asalnya sudah RGB dari main.py (yang dilempar ke detect_and_extract),
        # maka 'face_resized' otomatis sudah RGB murni. Jangan diconvert lagi!
        face_blob = face_resized.astype(np.float32) / 255.0
        face_blob = np.expand_dims(face_blob, axis=0) # Shape jadi (1, 112, 112, 3)

        # --- TAHAP 3: EKSTRAKSI VEKTOR 512 DENGAN GHOSTFACENETV2 ---
        ghost_outputs = self.ghost_session.run(None, {self.ghost_input_name: face_blob})
        embedding = ghost_outputs[0][0] # Ambil hasil vektornya
        
        # Lakukan L2 Normalization agar panjang vektornya bernilai 1.
        norm = np.linalg.norm(embedding)
        if norm > 0:
            embedding = embedding / norm
            
        # Mengembalikan list berisi 512 angka float reguler Python yang siap ditelan SQL
        return embedding.tolist()

# Inisialisasi satu kali secara global (Singleton) biar hemat RAM VPS lu saat dipanggil FastAPI
face_pipeline = FaceRecognitionPipeline()