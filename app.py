import streamlit as st
import cv2
import numpy as np
from ultralytics import YOLO
import time

# --- Page Configuration ---
st.set_page_config(page_title="Smart Assist AI Portal", layout="wide")
st.title("👁️ Smart Assist: Environmental Awareness Dashboard")

# --- Model Initialization ---
@st.cache_resource
def load_model():
    # Downloads and loads the model into memory once
    return YOLO("yolov8n.pt")

model = load_model()

# Your specific obstacle list
FOCUS_CLASSES = [
    'person', 'stairs', 'pothole', 'door', 'table', 
    'vehicle', 'plant', 'fence', 'garbage bin'
]

# --- Sidebar Controls ---
st.sidebar.header("System Settings")
run_engine = st.sidebar.checkbox("Launch Smart Assist Engine")
conf_val = st.sidebar.slider("Detection Confidence", 0.1, 1.0, 0.35)

# Metrics display slots
col1, col2, col3 = st.columns(3)
stat_count = col1.empty()
stat_light = col2.empty()
stat_health = col3.empty()

# The placeholder where the video feed will appear
FRAME_WINDOW = st.image([])

# --- Main Logic ---
if run_engine:
    cap = cv2.VideoCapture(0)
    while run_engine:
        ret, frame = cap.read()
        if not ret:
            st.error("Cannot access camera.")
            break

        # 1. Low Light Check & Enhancement
        brightness = np.mean(frame)
        if brightness < 80:
            lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
            l, a, b = cv2.split(lab)
            clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
            l = clahe.apply(l)
            frame = cv2.cvtColor(cv2.merge((l,a,b)), cv2.COLOR_LAB2BGR)

        # 2. Run Inference
        results = model(frame, conf=conf_val, verbose=False)[0]
        h, w = frame.shape[:2]
        active_detections = []

        # 3. Process and Annotate
        for r in results.boxes:
            c_id = int(r.cls[0])
            name = model.names[c_id].lower()
            
            # Filter for your specific obstacles
            if any(fc in name for fc in FOCUS_CLASSES):
                x1, y1, x2, y2 = r.xyxy[0]
                # Calculate closeness % and dimensions
                box_w, box_h = int(x2-x1), int(y2-y1)
                closeness = ((box_w * box_h) / (w * h)) * 100
                
                # Visuals
                color = (255, 0, 0) if closeness > 25 else (0, 255, 0) # Red if very close
                cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
                cv2.putText(frame, f"{name} {closeness:.1f}%", (int(x1), int(y1)-10), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                active_detections.append(name)

        # 4. Update UI Stats
        stat_count.metric("Active Obstacles", len(active_detections))
        stat_light.metric("Brightness", f"{int(brightness)}")
        stat_health.metric("Mode", "Night Vision" if brightness < 80 else "Standard")

        # 5. Convert to RGB for Streamlit display
        FRAME_WINDOW.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        
    cap.release()
else:
    st.info("System is in standby. Use the sidebar to start the camera.")