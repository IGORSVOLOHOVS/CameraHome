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
            sync_status TEXT DEFAULT 'pending'
        )
    """)
    conn.commit()
    conn.close()

def log_metrics(green_percentage, leaf_count, temperature=None, humidity=None):
    init_db()
    conn = sqlite3.connect(DB_FILE)
    cursor = conn.cursor()
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    cursor.execute("""
        INSERT INTO metrics (timestamp, green_percentage, leaf_count, temperature, humidity, sync_status)
        VALUES (?, ?, ?, ?, ?, 'pending')
    """, (timestamp, green_percentage, leaf_count, temperature, humidity))
    conn.commit()
    conn.close()
    return timestamp

def get_unsynced_metrics():
    init_db()
    conn = sqlite3.connect(DB_FILE)
    cursor = conn.cursor()
    cursor.execute("SELECT id, timestamp, green_percentage, leaf_count, temperature, humidity FROM metrics WHERE sync_status = 'pending'")
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
