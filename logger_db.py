import sqlite3
from datetime import datetime

DB_FILE = "plant_metrics.db"

def init_db():
    conn = sqlite3.connect(DB_FILE)
    cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS metrics (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT NOT NULL,
            green_percentage REAL NOT NULL,
            leaf_count INTEGER NOT NULL,
            temperature REAL,
            humidity REAL,
            photo_url TEXT,
            sync_status TEXT DEFAULT 'pending'
        )
    """)
    # Migration: add photo_url column if database already exists without it
    try:
        cursor.execute("ALTER TABLE metrics ADD COLUMN photo_url TEXT")
    except sqlite3.OperationalError:
        pass # Column already exists
    conn.commit()
    conn.close()

def log_metrics(green_percentage, leaf_count, temperature=None, humidity=None, photo_url=None):
    init_db()
    conn = sqlite3.connect(DB_FILE)
    cursor = conn.cursor()
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    cursor.execute("""
        INSERT INTO metrics (timestamp, green_percentage, leaf_count, temperature, humidity, photo_url, sync_status)
        VALUES (?, ?, ?, ?, ?, ?, 'pending')
    """, (timestamp, green_percentage, leaf_count, temperature, humidity, photo_url))
    conn.commit()
    conn.close()
    return timestamp

def get_unsynced_metrics():
    init_db()
    conn = sqlite3.connect(DB_FILE)
    cursor = conn.cursor()
    cursor.execute("SELECT id, timestamp, green_percentage, leaf_count, temperature, humidity, photo_url FROM metrics WHERE sync_status = 'pending'")
    rows = cursor.fetchall()
    conn.close()
    return rows

def mark_as_synced(ids):
    if not ids:
        return
    init_db()
    conn = sqlite3.connect(DB_FILE)
    cursor = conn.cursor()
    placeholders = ",".join("?" for _ in ids)
    cursor.execute(f"UPDATE metrics SET sync_status = 'synced' WHERE id IN ({placeholders})", ids)
    conn.commit()
    conn.close()
