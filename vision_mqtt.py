import os
import time
import json
import argparse
import requests
import numpy as np
import paho.mqtt.client as mqtt
from PIL import Image, ImageDraw

try:
    import tflite_runtime.interpreter as tflite
except ImportError:
    import tensorflow.lite as tflite

"""!
@brief Notification handler for Telegram.
"""
class TelegramNotifier:
    def __init__(self, token, chat_id=None):
        self.token = token
        self.chat_id = chat_id
        self.base_url = f"https://api.telegram.org/bot{token}"

    def get_chat_id(self):
        """!
        @brief Fetches Chat ID from the latest bot update.
        """
        try:
            r = requests.get(f"{self.base_url}/getUpdates", timeout=10).json()
            if r.get("ok") and r.get("result"):
                # Get Chat ID from the last message
                last_update = r["result"][-1]
                chat_id = last_update.get("message", {}).get("chat", {}).get("id")
                if chat_id:
                    self.chat_id = chat_id
                    print(f"[SUCCESS] Found Telegram Chat ID: {self.chat_id}")
                    return True
            print("[WARN] No messages found for the bot. Please send any message to your bot on Telegram!")
            return False
        except Exception as e:
            print(f"[ERROR] Chat discovery failed: {e}")
            return False

    def send_message(self, text):
        if not self.chat_id: return
        try:
            r = requests.post(f"{self.base_url}/sendMessage", data={"chat_id": self.chat_id, "text": text}, timeout=10).json()
            if not r.get("ok"):
                print(f"[ERROR] Telegram Message failed: {r.get('description')}")
            else:
                print("[DEBUG] Telegram Message sent!")
        except Exception as e:
            print(f"[ERROR] Telegram network error: {e}")

    def send_photo(self, photo_path, caption=None):
        if not self.chat_id: return
        try:
            with open(photo_path, 'rb') as photo:
                r = requests.post(f"{self.base_url}/sendPhoto", 
                              data={"chat_id": self.chat_id, "caption": caption},
                              files={"photo": photo}, timeout=10).json()
                if not r.get("ok"):
                    print(f"[ERROR] Telegram Photo failed: {r.get('description')}")
                else:
                    print("[DEBUG] Telegram Photo sent!")
        except Exception as e:
            print(f"[ERROR] Telegram network error: {e}")

"""!
@brief Main logic for Edge Vision MQTT on Termux.
@details Handles periodic snapshots and TFLite inference.
"""
class EdgeVision:
    def __init__(self, model_path, telegram=None, camera_id=0):
        self.interpreter = tflite.Interpreter(model_path=model_path)
        self.interpreter.allocate_tensors()
        self.mqtt = None
        self.telegram = telegram
        self.last_pub = 0
        self.topic = "home/camera/detection"
        self.snapshot_path = "snap.jpg"
        self.camera_id = camera_id

    def setup_net(self, host, port=1883):
        if not host:
            print("[INFO] MQTT Host not provided, skipping MQTT.")
            return

        try:
            self.mqtt = mqtt.Client()
            self.mqtt.connect(host, port, 60)
            self.mqtt.loop_start()
            print(f"[SUCCESS] Connected to MQTT at {host}:{port}")
        except Exception as e:
            print(f"[ERROR] MQTT failed: {e}")
            self.mqtt = None

    def process_frame(self):
        print("\n[DEBUG] Capturing frame...", end=" ", flush=True)
        if os.path.exists(self.snapshot_path): os.remove(self.snapshot_path)
        
        # Capture photo using termux-api with a timeout
        try:
            import subprocess
            cmd = ["termux-camera-photo", "-c", str(getattr(self, 'camera_id', 0)), self.snapshot_path]
            # Redirect stdout/stderr to devnull
            subprocess.run(cmd, timeout=15, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        except subprocess.TimeoutExpired:
            print("FAILED (Timeout). Another app might be using the camera or permission was not granted!")
            return None
        except Exception as e:
            print(f"FAILED (Error): {e}")
            return None
        
        if not os.path.exists(self.snapshot_path) or os.path.getsize(self.snapshot_path) == 0:
            print("FAILED (empty photo). Check permissions or camera ID! Try running 'termux-camera-info' to see available IDs.")
            return None
        
        try:
            # Open captured photo for TFLite
            orig_img = Image.open(self.snapshot_path)
            
            # 1. Resize for TFLite (YOLOv8 usually expects 640x640)
            tflite_img = orig_img.resize((640, 640))
            input_data = np.expand_dims(np.array(tflite_img, dtype=np.float32) / 255.0, axis=0)
            
            self.interpreter.set_tensor(self.interpreter.get_input_details()[0]['index'], input_data)
            self.interpreter.invoke()
            
            output = self.interpreter.get_tensor(self.interpreter.get_output_details()[0]['index'])[0]
            # output shape: [84, 8400] - Rows: cx, cy, w, h, class0, class1...
            person_scores = output[4]
            best_idx = np.argmax(person_scores)
            max_person_conf = person_scores[best_idx]
            
            # If detected, draw box on the original image
            if max_person_conf > 0.3:
                cx, cy, w, h = output[0:4, best_idx]
                
                # Robust scaling: Check if coords are normalized (0-1) or pixels (0-640)
                scale_factor = 1.0 if cx <= 1.0 else (1.0 / 640.0)
                
                img_w, img_h = orig_img.size
                x1 = (cx - w/2) * scale_factor * img_w
                y1 = (cy - h/2) * scale_factor * img_h
                x2 = (cx + w/2) * scale_factor * img_w
                y2 = (cy + h/2) * scale_factor * img_h
                
                print(f"[DEBUG] Box (orig): {x1:.0f},{y1:.0f} to {x2:.0f},{y2:.0f}")
                
                # Draw green rectangle - triple thickness for "big borders"
                draw = ImageDraw.Draw(orig_img)
                box_color = "green"
                draw.rectangle([x1, y1, x2, y2], outline=box_color, width=24)
                
                # Confidence label above the box
                label = f"PERSON {max_person_conf:.2f}"
                # Background for text (makes it readable)
                draw.rectangle([x1, y1 - 60, x1 + 450, y1], fill=box_color)
                draw.text((x1 + 10, y1 - 55), label, fill="white")
                
                # Overwrite the captured photo with the annotated one
                orig_img.save(self.snapshot_path)

            print(f"DONE (Conf: {max_person_conf:.2f})")
            return max_person_conf
        except Exception as e:
            print(f"ERROR (Invalid Photo): {e}")
            return None

    def run(self, threshold=0.6, cooldown=30):
        print(f"[START] Monitoring... Threshold: {threshold}, Cooldown: {cooldown}s")
        while True:
            conf = self.process_frame()
            now = time.time()
            
            if conf and conf > threshold:
                timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
                msg = f"[{timestamp}] PERSON DETECTED! (Conf: {conf:.2f})"
                print(f"!!! {msg} !!!")
                
                # Notification logic with cooldown
                if (now - self.last_pub > cooldown):
                    # MQTT
                    if self.mqtt:
                        self.mqtt.publish(self.topic, json.dumps({"status": "detected", "confidence": float(conf)}))
                    
                    # Telegram
                    if self.telegram:
                        print("[DEBUG] Compressing and sending Telegram notification...")
                        if not self.telegram.chat_id:
                            self.telegram.get_chat_id()
                        
                        try:
                            # 2. Compress ONLY when sending (640px, quality 70 for speed)
                            orig_img = Image.open(self.snapshot_path)
                            orig_img.thumbnail((640, 640), Image.Resampling.LANCZOS)
                            orig_img.save("snap_tg.jpg", "JPEG", quality=70)
                            
                            self.telegram.send_photo("snap_tg.jpg", caption=msg)
                        except Exception as e:
                            print(f"[ERROR] Failed to compress/send: {e}")
                    
                    self.last_pub = now
            
            time.sleep(1)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Edge Vision on Termux")
    parser.add_argument("--model", default="yolov8n_float16.tflite", help="Path to TFLite model")
    parser.add_argument("--host", help="MQTT Broker IP address")
    parser.add_argument("--token", default="REDACTED_TOKEN", help="Telegram Bot Token")
    parser.add_argument("--chat_id", help="Telegram Chat ID (optional)")
    parser.add_argument("--threshold", type=float, default=0.6, help="Detection threshold")
    parser.add_argument("--camera", type=int, default=0, help="Camera ID to use (check termux-camera-info)")
    parser.add_argument("--cooldown", type=int, default=30, help="Cooldown between notifications (seconds)")
    
    args = parser.parse_args()

    tg = TelegramNotifier(args.token, args.chat_id) if args.token else None
    vision = EdgeVision(args.model, telegram=tg, camera_id=args.camera)
    vision.setup_net(args.host)
    vision.run(threshold=args.threshold, cooldown=args.cooldown)
