import streamlit as st
import cv2
import numpy as np
from ultralytics import YOLO
import pyttsx3
import threading

# --- Page Layout Setup ---
st.set_page_config(page_title="Smart Assist Dashboard", layout="wide", page_icon="👁️")
st.title("👁️ Smart Assist: Environmental Awareness Portal")
st.write("Development Stage: Local Hardware Prototype Interface")

# --- Initialize Text-to-Speech Engine ---
@st.cache_resource
def init_tts():
    engine = pyttsx3.init()
    engine.setProperty('rate', 160)  # Moderate speaking speed
    return engine

tts_engine = init_tts()

def speak_alert(text):
    """Runs audio alerts in a background thread to prevent the video feed from lagging."""
    def target():
        try:
            tts_engine.say(text)
            tts_engine.runAndWait()
        except:
            pass
    threading.Thread(target=target, daemon=True).start()

# --- Load YOLOv8 Model ---
@st.cache_resource
def load_model():
    return YOLO("yolov8n.pt")

model = load_model()

# Core obstacle classes for assistive navigation
FOCUS_CLASSES = ['person', 'stairs', 'door', 'table', 'chair', 'vehicle', 'pothole']

# --- UI Controls Sidebar ---
st.sidebar.header("🔧 System Calibration")
confidence_threshold = st.sidebar.slider("Detection Confidence", 0.1, 1.0, 0.25)

st.sidebar.markdown("""
### 💡 Viva Presentation Tip
* **CLAHE Enhancement:** Applied dynamically below a mean brightness of 80 to ensure spatial clarity in low-light environments.
* **Proximity Metric:** Bounding box pixel density percentage relative to the canvas calculation determines hazard severity.
""")

# Main toggle checkbox to kick off webcam capture
run_engine = st.checkbox("Launch Smart Assist Webcam Engine")
FRAME_WINDOW = st.image([])  # Streamlit placeholder container for the video feed

# --- Core Processing Matrix ---
if run_engine:
    cap = cv2.VideoCapture(0)  # Access local laptop webcam
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            st.error("Hardware Alert: Webcam stream disconnected.")
            break
            
        # 1. Low Light Processing Module (CLAHE)
        if np.mean(frame) < 80:
            lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
            l, a, b = cv2.split(lab)
            clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
            l = clahe.apply(l)
            frame = cv2.cvtColor(cv2.merge((l, a, b)), cv2.COLOR_LAB2BGR)

        # 2. Machine Learning Inference Pipeline
        results = model(frame, conf=confidence_threshold, verbose=False)[0]
        h, w = frame.shape[:2]
        
        highest_hazard = None
        max_proximity = 0
        
        for r in results.boxes:
            c_id = int(r.cls[0])
            name = model.names[c_id].lower()
            
            if any(fc in name for fc in FOCUS_CLASSES):
                x1, y1, x2, y2 = r.xyxy[0]
                
                # Proximity math: (Box Area / Total Image Area) * 100
                proximity_score = (((x2 - x1) * (y2 - y1)) / (w * h)) * 100
                
                # Determine box color layout
                if proximity_score > 25:
                    color = (0, 0, 255)  # Red warning boundary
                    if proximity_score > max_proximity:
                        max_proximity = proximity_score
                        highest_hazard = name
                else:
                    color = (0, 255, 0)  # Safe green boundary
                    
                # Render visual boundaries directly onto frame matrix
                cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
                cv2.putText(frame, f"{name} {proximity_score:.1f}%", (int(x1), int(y1) - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        
        # 3. Fire Voice Warnings Safely
        if highest_hazard:
            speak_alert(f"Warning. {highest_hazard} ahead.")
            
        # 4. Refresh Web UI Viewport
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        FRAME_WINDOW.image(frame_rgb)
        
    cap.release()
else:
    st.info("System Standby. Toggle the engine checkbox to boot up the vision tracking matrix.")
