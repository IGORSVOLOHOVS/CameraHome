import os

def load_env(path=".env"):
    if not os.path.exists(path):
        return
    with open(path) as f:
        for line in f:
            line = line.strip()
            if "=" in line and not line.startswith("#"):
                k, v = line.split("=", 1)
                os.environ[k.strip()] = v.strip()

# Load env variables on import
load_env()

# Bot & API Credentials
TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID")
SPREADSHEET_ID = os.getenv("SPREADSHEET_ID")
CREDENTIALS_FILE = os.getenv("CREDENTIALS_FILE", "credentials.json")

# App Settings
CAMERA_ID = int(os.getenv("CAMERA_ID", "0"))
MODEL_PATH = os.getenv("MODEL_PATH", "yolov8n_float16.tflite")
CONFIDENCE_THRESHOLD = float(os.getenv("THRESHOLD", "0.25"))
COOLDOWN = int(os.getenv("COOLDOWN", "30"))

# Scheduler
INTERVAL_HOURS = float(os.getenv("INTERVAL_HOURS", "3"))
SIMULATION = os.getenv("SIMULATION", "False").lower() in ("true", "1", "yes")

# MQTT (For future expansion)
MQTT_HOST = os.getenv("MQTT_HOST")
MQTT_PORT = int(os.getenv("MQTT_PORT", "1883"))
MQTT_TOPIC = os.getenv("MQTT_TOPIC", "home/plant/metrics")
