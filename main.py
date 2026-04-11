import cv2
import numpy as np
import supervision as sv
import pyttsx3
import threading
import time
from ultralytics import YOLO

# --- 1. Initialization ---
# (We handle the main engine initialization inside the speak function now)
low_light_announced = False 

def speak(text):
    """Thread-safe non-blocking TTS."""
    def run():
        try:
            local_engine = pyttsx3.init()
            local_engine.setProperty('rate', 170)
            local_engine.say(text)
            local_engine.runAndWait()
            local_engine.stop()
        except:
            pass
    threading.Thread(target=run, daemon=True).start()

# Load models
model_gen = YOLO("yolov8n.pt") 
model_haz = YOLO("yolov8n.pt") 

def process_frame(frame):
    global low_light_announced
    
    # 2. Camera Focus/Sharpness Assessment
    focus_measure = cv2.Laplacian(frame, cv2.CV_64F).var()
    
    # 3. Brightness Assessment
    avg_color = np.mean(frame, axis=(0, 1))
    brightness = np.mean(avg_color) 
    is_dark = brightness < 80

    # 4. Conditional Contrast Enhancement (CLAHE)
    processing_image = frame.copy()
    dark_warning = ""
    
    if is_dark:
        if not low_light_announced:
            dark_warning = "Low light detected. Switching to enhanced vision."
            low_light_announced = True
        
        lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        l = clahe.apply(l)
        processing_image = cv2.cvtColor(cv2.merge((l, a, b)), cv2.COLOR_LAB2BGR)
    else:
        low_light_announced = False

    # 5. Parallel Detection Threads
    det_gen, det_haz = [], []

    def run_gen():
        nonlocal det_gen
        results = model_gen(processing_image, conf=0.35, verbose=False)[0]
        det_gen = sv.Detections.from_ultralytics(results)

    def run_haz():
        nonlocal det_haz
        results = model_haz(processing_image, conf=0.40, verbose=False)[0]
        det_haz = sv.Detections.from_ultralytics(results)

    t1 = threading.Thread(target=run_gen)
    t2 = threading.Thread(target=run_haz)
    t1.start(); t2.start()
    t1.join(); t2.join()

    # 6. Combine Detections
    combined_detections = sv.Detections.merge([det_gen, det_haz])
    
    # 7. Camera Health Check Logic (Adjusted threshold to 60.0)
    camera_is_blur = focus_measure < 60.0 
    camera_alert = "Camera view is unclear" if camera_is_blur else ""

    # 8. Distance & Direction Logic
    alert_parts = []
    trigger_beep = False
    img_h, img_w = frame.shape[:2]

    for i in range(len(combined_detections)):
        x1, y1, x2, y2 = combined_detections.xyxy[i]
        norm_area = ((x2 - x1) * (y2 - y1)) / (img_w * img_h)
        
        dist_label = "far"
        if norm_area > 0.25:
            dist_label = "very close"
            trigger_beep = True
        elif norm_area > 0.08:
            dist_label = "near"

        center_x = (x1 + x2) / 2 / img_w
        if center_x < 0.35: dir_label = "to the left"
        elif center_x > 0.65: dir_label = "to the right"
        else: dir_label = "in front"

        class_name = model_gen.names[combined_detections.class_id[i]]
        alert_parts.append(f"{class_name} {dist_label} {dir_label}")

    # 9. Visualization
    annotator = sv.BoxAnnotator()
    label_annotator = sv.LabelAnnotator()
    annotated_frame = annotator.annotate(scene=processing_image, detections=combined_detections)
    annotated_frame = label_annotator.annotate(scene=annotated_frame, detections=combined_detections)

    detection_msg = ", ".join(alert_parts)
    return detection_msg, trigger_beep, annotated_frame, dark_warning, camera_alert

# --- 10. Main Execution Loop ---
cap = cv2.VideoCapture(0)
last_alert_time = 0
last_blur_time = 0

print("--- Smart Assist Workflow Started ---")

while cap.isOpened():
    ret, frame = cap.read()
    if not ret: break

    msg, beep, visual, dark_msg, blur_msg = process_frame(frame)
    current_time = time.time()

    # 1. Darkness Warning
    if dark_msg:
        speak(dark_msg)
        last_alert_time = current_time 

    # 2. Camera Blur Alert
    # We use a 7-second cooldown so it's not too repetitive
    if blur_msg and (current_time - last_blur_time > 7.0):
        print(f"!!! SYSTEM ALERT: {blur_msg} !!!")
        speak(blur_msg)
        last_blur_time = current_time

    # 3. Object Detections
    elif msg and (current_time - last_alert_time > 3.0):
        print(f"User Feedback: {msg}")
        speak(msg)
        last_alert_time = current_time
    
    if beep:
        print("\a") 

    cv2.imshow("Smart Assist Monitor", visual)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()