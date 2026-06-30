import os
import psycopg2
from psycopg2.extras import RealDictCursor
from dotenv import load_dotenv

# Load file .env untuk mengambil rahasia dapur koneksi database
load_dotenv()

DATABASE_URL = os.getenv("DATABASE_URL")

if not DATABASE_URL:
    raise ValueError("database is missing")

def get_db_connection():
    """
    try to connect
    """
    try:
        conn = psycopg2.connect(DATABASE_URL, cursor_factory=RealDictCursor)
        return conn
    except Exception as e:
        print(f"connection failed: {e}")
        raise e

def init_db():
    conn = get_db_connection()
    cur = conn.cursor()
    try:
        # 1. Pastikan ekstensi pgvector menyala
        cur.execute("CREATE EXTENSION IF NOT EXISTS vector;")
        
        # 2. Bikin tabel dengan skema UUID Aman
        # Kolom id sekarang diganti dari SERIAL menjadi VARCHAR(50) PRIMARY KEY
        cur.execute("""
            CREATE TABLE IF NOT EXISTS users (
                id VARCHAR(50) PRIMARY KEY,
                name VARCHAR(100) NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );
            
            CREATE TABLE IF NOT EXISTS face_features (
                id SERIAL PRIMARY KEY,
                user_id VARCHAR(50) REFERENCES users(id) ON DELETE CASCADE,
                face_embedding vector(512) NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );
            
            CREATE INDEX IF NOT EXISTS face_features_cosine_idx 
            ON face_features USING hnsw (face_embedding vector_cosine_ops);
        """)
        
        conn.commit()
        print("ready: database initialized successfully!")
    except Exception as e:
        conn.rollback()
        print(f"failed: {e}")
    finally:
        cur.close()
        conn.close()

# Iseng panggil otomatis pas file ini di-load pertama kali biar lu gak lupa mengaktifkan fiturnya
init_db()