import os
import time
import json
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
@see TFLite Guide (https://www.tensorflow.org/lite/guide/python)
"""
class EdgeVision:
    """!
    @brief Core class for surveillance automation.
    @param[in] model_path Path to local .tflite file.
    """
    def __init__(self, model_path):
        self.interpreter = tflite.Interpreter(model_path=model_path)
        self.interpreter.allocate_tensors()
        self.mqtt = mqtt.Client()
        self.last_pub = 0
        self.topic = "home/camera/detection"

    """!
    @brief Connects to MQTT broker.
    @param[in] host Broker IP address.
    """
    def setup_net(self, host):
        self.mqtt.connect(host, 1883, 60)
        self.mqtt.loop_start()

    """!
    @brief Captures frame and runs YOLOv8 detection.
    @return float Maximum confidence score for class 0 (person).
    """
    def process_frame(self):
        os.system("termux-camera-photo -c 0 snap.jpg")
        if not os.path.exists("snap.jpg"): return None
        img = Image.open("snap.jpg").resize((320, 320))
        input_data = np.expand_dims(np.array(img, dtype=np.float32) / 255.0, axis=0)
        self.interpreter.set_tensor(self.interpreter.get_input_details()[0]['index'], input_data)
        self.interpreter.invoke()
        output = self.interpreter.get_tensor(self.interpreter.get_output_details()[0]['index'])[0]
        # YOLOv8 output: [1, 84, 8400] -> Index 4 is Class 0 (Person)
        max_person_conf = np.max(output[4])
        return max_person_conf if max_person_conf > 0.6 else None

    """!
    @brief Main detection loop.
    @note Runs at ~1 FPS to save battery.
    """
    def run(self):
        while True:
            conf = self.process_frame()
            now = time.time()
            if conf and (now - self.last_pub > 5):
                self.mqtt.publish(self.topic, json.dumps({"status": "detected", "confidence": float(conf)}))
                self.last_pub = now
            time.sleep(1)

if __name__ == "__main__":
    vision = EdgeVision("yolo.tflite")
    vision.setup_net("192.168.1.100")
    vision.run()
