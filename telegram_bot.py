import threading
import time
import requests
import os
import config
import cv_yolo
import logger_db
import updater

class TelegramBot:
    def __init__(self, token, chat_id=None, analyzer=None, on_simulate_toggle=None):
        self.token = token
        self.chat_id = chat_id
        self.analyzer = analyzer
        self.on_simulate_toggle = on_simulate_toggle
        self.base_url = f"https://api.telegram.org/bot{token}"
        self.offset = 0
        self.running = False
        self.thread = None

    def send_message(self, text):
        if not self.chat_id:
            return
        try:
            requests.post(f"{self.base_url}/sendMessage", data={"chat_id": self.chat_id, "text": text}, timeout=10)
        except Exception as e:
            print(f"[ERROR] Failed to send message: {e}")

    def send_photo(self, photo_path, caption=None):
        if not self.chat_id:
            return
        try:
            from PIL import Image
            # Compress image to prevent network timeouts
            img = Image.open(photo_path)
            img.thumbnail((1024, 1024), Image.Resampling.LANCZOS)
            temp_path = "snap_compressed.jpg"
            img.save(temp_path, "JPEG", quality=75)
            
            with open(temp_path, 'rb') as photo:
                r = requests.post(f"{self.base_url}/sendPhoto", 
                                  data={"chat_id": self.chat_id, "caption": caption},
                                  files={"photo": photo}, timeout=25)
                if not r.json().get("ok"):
                    print(f"[ERROR] Telegram photo send failed: {r.text}")
                    
            if os.path.exists(temp_path):
                os.remove(temp_path)
        except Exception as e:
            print(f"[ERROR] Failed to send photo: {e}")

    def get_updates(self):
        try:
            url = f"{self.base_url}/getUpdates?offset={self.offset}&timeout=5"
            r = requests.get(url, timeout=10).json()
            if r.get("ok"):
                return r.get("result", [])
        except Exception as e:
            print(f"[ERROR] Failed to get Telegram updates: {e}")
        return []

    def handle_command(self, text):
        cmd = text.strip().lower()
        if cmd.startswith("/start") or cmd.startswith("/help"):
            help_text = (
                "🌱 *Plant Monitor Bot* 🌱\n\n"
                "Available Commands:\n"
                "/photo - Take a photo and analyze it immediately\n"
                "/status - Get current system status and last metrics\n"
                "/simulate - Toggle simulation mode (1 min interval vs 3 hours)\n"
                "/update - Check for updates on GitHub and self-restart"
            )
            self.send_message(help_text)
            
        elif cmd.startswith("/photo"):
            self.send_message("📸 Capturing and analyzing frame, please wait...")
            if self.analyzer:
                success = self.analyzer.capture_photo("snap.jpg")
                if success:
                    green_pct, leaf_cnt = self.analyzer.analyze_image("snap.jpg")
                    caption = f"📊 Analysis Results:\n- Green Mass: {green_pct:.2f}%\n- Plants/Leaves detected: {leaf_cnt}"
                    self.send_photo("snap.jpg", caption=caption)
                else:
                    self.send_message("❌ Failed to capture photo. The camera might be busy.")
            else:
                self.send_message("❌ Plant analyzer module is not initialized.")
                
        elif cmd.startswith("/status"):
            # Get last metrics from DB
            unsynced = logger_db.get_unsynced_metrics()
            mode = "Simulation (1 min)" if config.SIMULATION else f"Production ({config.INTERVAL_HOURS} hours)"
            status_msg = (
                f"⚙️ *System Status*:\n"
                f"- Mode: {mode}\n"
                f"- Unsynced cache: {len(unsynced)} records\n"
                f"- YOLO Model: {os.path.basename(config.MODEL_PATH)}\n"
                f"- Camera ID: {config.CAMERA_ID}\n"
            )
            self.send_message(status_msg)
            
        elif cmd.startswith("/simulate"):
            config.SIMULATION = not config.SIMULATION
            mode_str = "Enabled" if config.SIMULATION else "Disabled"
            self.send_message(f"🔄 Simulation mode {mode_str}. Re-initializing scheduler...")
            if self.on_simulate_toggle:
                self.on_simulate_toggle()
                
        elif cmd.startswith("/update"):
            self.send_message("🔄 Checking GitHub for updates...")
            updated, msg = updater.check_and_pull()
            self.send_message(msg)
            if updated:
                self.send_message("🚀 Restarting script to apply updates...")
                time.sleep(1)
                updater.restart_script()
        else:
            self.send_message("Unknown command. Type /help to see available commands.")

    def poll_loop(self):
        print("[INFO] Telegram Bot polling started.")
        # Perform auto-discovery of chat_id if not configured
        if not self.chat_id:
            print("[INFO] No TELEGRAM_CHAT_ID configured. Waiting for any message to start...")
            while not self.chat_id and self.running:
                updates = self.get_updates()
                for upd in updates:
                    self.offset = upd["update_id"] + 1
                    msg = upd.get("message")
                    if msg:
                        self.chat_id = msg["chat"]["id"]
                        print(f"[SUCCESS] Auto-discovered Chat ID: {self.chat_id}")
                        self.send_message("👋 Hello! Chat linked successfully. Bot is ready.")
                        break
                time.sleep(2)

        while self.running:
            updates = self.get_updates()
            for upd in updates:
                self.offset = upd["update_id"] + 1
                msg = upd.get("message")
                if msg and msg.get("text"):
                    # Check if command is from authorized user (optional, but good practice)
                    if str(msg["chat"]["id"]) == str(self.chat_id):
                        self.handle_command(msg["text"])
            time.sleep(1)

    def start(self):
        if self.running:
            return
        self.running = True
        self.thread = threading.Thread(target=self.poll_loop, daemon=True)
        self.thread.start()

    def stop(self):
        self.running = False
        if self.thread:
            self.thread.join(timeout=3)
