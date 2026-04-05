import os
import time
import json
import argparse
import numpy as np
import paho.mqtt.client as mqtt
from PIL import Image

try:
    import tflite_runtime.interpreter as tflite
except ImportError:
    import tensorflow.lite as tflite

"""!
@brief Main logic for Edge Vision MQTT on Termux.
@details Handles periodic snapshots and TFLite inference.
"""
class EdgeVision:
    """!
    @brief Core class for surveillance automation.
    @param[in] model_path Path to local .tflite file.
    """
    def __init__(self, model_path):
        self.interpreter = tflite.Interpreter(model_path=model_path)
        self.interpreter.allocate_tensors()
        self.mqtt = None
        self.last_pub = 0
        self.topic = "home/camera/detection"

    """!
    @brief Connects to MQTT broker.
    @param[in] host Broker IP address.
    @param[in] port Broker Port.
    """
    def setup_net(self, host, port=1883):
        if not host:
            print("[INFO] MQTT Host not provided, running in local-only mode.")
            return

        try:
            self.mqtt = mqtt.Client()
            self.mqtt.connect(host, port, 60)
            self.mqtt.loop_start()
            print(f"[SUCCESS] Connected to MQTT broker at {host}:{port}")
        except Exception as e:
            print(f"[ERROR] Failed to connect to MQTT: {e}")
            self.mqtt = None

    """!
    @brief Captures frame and runs YOLOv8 detection.
    @return float Maximum confidence score for class 0 (person).
    """
    def process_frame(self):
        # Capture photo using Termux API
        os.system("termux-camera-photo -c 0 snap.jpg")
        if not os.path.exists("snap.jpg"):
            print("[WARN] snap.jpg not found, skipping frame.")
            return None
        
        try:
            img = Image.open("snap.jpg").resize((320, 320))
            input_data = np.expand_dims(np.array(img, dtype=np.float32) / 255.0, axis=0)
            
            self.interpreter.set_tensor(self.interpreter.get_input_details()[0]['index'], input_data)
            self.interpreter.invoke()
            
            output = self.interpreter.get_tensor(self.interpreter.get_output_details()[0]['index'])[0]
            # YOLOv8 output: [1, 84, 8400] -> Index 4 is Class 0 (Person)
            max_person_conf = np.max(output[4])
            return max_person_conf
        except Exception as e:
            print(f"[ERROR] Detection failed: {e}")
            return None

    """!
    @brief Main detection loop.
    """
    def run(self, threshold=0.6):
        print(f"[START] Monitoring... Threshold: {threshold}")
        while True:
            conf = self.process_frame()
            now = time.time()
            
            if conf and conf > threshold:
                timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
                print(f"[{timestamp}] PERSON DETECTED! Confidence: {conf:.2f}")
                
                # Publish to MQTT if connected
                if self.mqtt and (now - self.last_pub > 5):
                    self.mqtt.publish(self.topic, json.dumps({"status": "detected", "confidence": float(conf)}))
                    self.last_pub = now
            
            time.sleep(1)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Edge Vision MQTT on Termux")
    parser.add_argument("--model", default="yolo.tflite", help="Path to TFLite model")
    parser.add_argument("--host", help="MQTT Broker IP address")
    parser.add_argument("--port", type=int, default=1883, help="MQTT Broker port")
    parser.add_argument("--threshold", type=float, default=0.6, help="Detection threshold (0.0-1.0)")
    
    args = parser.parse_args()

    vision = EdgeVision(args.model)
    vision.setup_net(args.host, args.port)
    vision.run(threshold=args.threshold)
