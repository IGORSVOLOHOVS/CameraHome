import time
import os
import sys
from datetime import datetime
import config
import cv_yolo
import logger_db
import google_sheets
from telegram_bot import TelegramBot

# Initialize database
logger_db.init_db()

analyzer = PlantAnalyzer = cv_yolo.PlantAnalyzer(config.MODEL_PATH)
bot = None
last_measurement_time = 0

def run_measurement():
    global last_measurement_time
    now = time.time()
    
    print(f"\n[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Starting periodic measurement...")
    
    success = analyzer.capture_photo("snap.jpg")
    if not success:
        msg = "⚠️ Failed to capture photo for periodic measurement!"
        print(msg)
        if bot:
            bot.send_message(msg)
        return False
        
    green_pct, leaf_cnt = analyzer.analyze_image("snap.jpg")
    
    # Write to local DB
    timestamp = logger_db.log_metrics(
        green_percentage=green_pct,
        leaf_count=leaf_cnt,
        temperature=None, # In future, read from MQTT
        humidity=None     # In future, read from MQTT
    )
    
    msg = (
        f"📊 *Periodic measurement complete*:\n"
        f"- Time: {timestamp}\n"
        f"- Green mass: {green_pct:.2f}%\n"
        f"- Plants/Leaves: {leaf_cnt}"
    )
    print(msg)
    
    # Send snapshot to Telegram
    if bot:
        bot.send_photo("snap.jpg", caption=msg)
        
    # Sync with Google Sheets
    sync_success, rows = google_sheets.sync_to_sheets()
    if sync_success and rows > 0:
        sync_msg = f"✅ Synced {rows} rows to Google Sheets."
        print(sync_msg)
        if bot:
            bot.send_message(sync_msg)
            
    last_measurement_time = now
    return True

def handle_simulation_toggle():
    global last_measurement_time
    # Reset last measurement time so it triggers immediately
    last_measurement_time = 0
    print(f"[INFO] Simulation mode toggled. Current state: {config.SIMULATION}")

def main():
    global bot, last_measurement_time
    
    print("=" * 50)
    print("🌱 PLANT GROWTH MONITORING FOR TERMUX STARTING 🌱")
    print("=" * 50)
    
    # Setup Telegram Bot
    if not config.TELEGRAM_TOKEN:
        print("[ERROR] TELEGRAM_TOKEN not set in .env! Cannot start Telegram bot.")
        sys.exit(1)
        
    bot = TelegramBot(
        token=config.TELEGRAM_TOKEN,
        chat_id=config.TELEGRAM_CHAT_ID,
        analyzer=analyzer,
        on_simulate_toggle=handle_simulation_toggle
    )
    bot.start()
    
    # Send startup notification
    startup_msg = "🚀 Plant Growth Monitor has started on Termux!"
    bot.send_message(startup_msg)
    
    print("[INFO] Scheduler loop running. Press Ctrl+C to exit.")
    try:
        while True:
            now = time.time()
            # Calculate interval (in seconds)
            if config.SIMULATION:
                interval = 60  # 1 minute in simulation mode
            else:
                interval = config.INTERVAL_HOURS * 3600  # Default 3 hours
                
            if now - last_measurement_time >= interval:
                run_measurement()
                
            time.sleep(1)
    except KeyboardInterrupt:
        print("\n[INFO] Stopping service...")
    finally:
        if bot:
            bot.stop()
        print("[INFO] Termux Plant Monitor Stopped.")

if __name__ == "__main__":
    main()
