import os
import subprocess
import numpy as np
import cv2
from PIL import Image
import config

try:
    import tflite_runtime.interpreter as tflite
except ImportError:
    import tensorflow.lite as tflite

class PlantAnalyzer:
    def __init__(self, model_path=config.MODEL_PATH):
        self.model_path = model_path
        self.interpreter = None
        if os.path.exists(model_path):
            try:
                self.interpreter = tflite.Interpreter(model_path=model_path)
                self.interpreter.allocate_tensors()
                print(f"[SUCCESS] Loaded YOLO TFLite model from {model_path}")
            except Exception as e:
                print(f"[ERROR] Failed to load TFLite model: {e}")
        else:
            print(f"[WARN] TFLite model file not found at {model_path}")

    def capture_photo(self, output_path="snap.jpg"):
        if os.path.exists(output_path):
            os.remove(output_path)
        try:
            cmd = ["termux-camera-photo", "-c", str(config.CAMERA_ID), output_path]
            subprocess.run(cmd, timeout=15, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        except subprocess.TimeoutExpired:
            print("[ERROR] Camera photo capture timed out!")
            return False
        except Exception as e:
            print(f"[ERROR] Camera photo capture failed: {e}")
            return False
        
        return os.path.exists(output_path) and os.path.getsize(output_path) > 0

    def analyze_image(self, img_path="snap.jpg", yolo_class_id=58):
        """
        Analyzes the image for green sprouts.
        Returns:
            green_percentage: float (0-100) based on HSV green color masking
            leaf_count: int, count of YOLO detections for the target plant class
        """
        img = cv2.imread(img_path)
        if img is None:
            print(f"[ERROR] Could not read image at {img_path}")
            return 0.0, 0

        h_img, w_img = img.shape[:2]
        total_pixels = h_img * w_img

        # 1. HSV Green Color Segmentation (highly accurate for tracking growth volume)
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        # Define range of green color in HSV
        lower_green = np.array([35, 40, 40])
        upper_green = np.array([85, 255, 255])
        green_mask = cv2.inRange(hsv, lower_green, upper_green)
        green_pixels = cv2.countNonZero(green_mask)
        green_percentage = (green_pixels / total_pixels) * 100.0
        # Outline green areas with bright green contours (borders)
        contours, _ = cv2.findContours(green_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for cnt in contours:
            if cv2.contourArea(cnt) > 50:  # Filter small noise
                cv2.drawContours(img, [cnt], -1, (0, 255, 0), 3)  # Green contour, thickness 3

        leaf_count = 0

        # 2. YOLO Neural Network Detection
        if self.interpreter:
            try:
                # Prepare image for YOLOv8 (640x640, float32, normalized to 0-1)
                img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                img_resized = cv2.resize(img_rgb, (640, 640))
                input_data = np.expand_dims(img_resized.astype(np.float32) / 255.0, axis=0)

                input_details = self.interpreter.get_input_details()
                output_details = self.interpreter.get_output_details()

                self.interpreter.set_tensor(input_details[0]['index'], input_data)
                self.interpreter.invoke()

                # YOLOv8 output is usually shape [1, 84, 8400]
                output = self.interpreter.get_tensor(output_details[0]['index'])[0]
                
                # Class 58 is potted plant in COCO, but we can search for it or any target class.
                # output row 4 to end are class scores
                num_classes = output.shape[0] - 4
                
                # We extract target class scores
                # If target class is outside the model's range, default to first class
                target_class = yolo_class_id if yolo_class_id < num_classes else 0
                scores = output[4 + target_class]
                
                # Find detections above threshold
                detect_indices = np.where(scores > config.CONFIDENCE_THRESHOLD)[0]
                
                # Apply Non-Maximum Suppression (NMS) to avoid double detections
                boxes = []
                confidences = []
                for idx in detect_indices:
                    cx, cy, w, h = output[0:4, idx]
                    # Scale coordinates back to original image dimensions
                    scale_factor = 1.0 if cx <= 1.0 else (1.0 / 640.0)
                    
                    x1 = int((cx - w / 2) * scale_factor * w_img)
                    y1 = int((cy - h / 2) * scale_factor * h_img)
                    box_w = int(w * scale_factor * w_img)
                    box_h = int(h * scale_factor * h_img)
                    
                    boxes.append([x1, y1, box_w, box_h])
                    confidences.append(float(scores[idx]))
                
                indices = cv2.dnn.NMSBoxes(boxes, confidences, config.CONFIDENCE_THRESHOLD, 0.45)
                
                if len(indices) > 0:
                    # If indices is 2D (like older OpenCV versions), flatten it
                    if isinstance(indices, np.ndarray):
                        indices = indices.flatten()
                    
                    leaf_count = len(indices)
                    for i in indices:
                        x, y, w, h = boxes[i]
                        conf = confidences[i]
                        
                        # Draw bounding box
                        cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 4) # Blue for YOLO bounding box
                        label = f"Plant: {conf:.2f}"
                        cv2.putText(img, label, (x, max(30, y - 10)), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 0, 0), 2)
                        
            except Exception as e:
                print(f"[ERROR] YOLO analysis failed: {e}")

        # Save annotated image
        cv2.imwrite(img_path, img)
        return green_percentage, leaf_count
