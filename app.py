import streamlit as st
from streamlit_webrtc import webrtc_streamer, WebRtcMode, RTCConfiguration
import av
import cv2
import numpy as np
from ultralytics import YOLO

# --- Page Setup ---
st.set_page_config(page_title="Smart Assist AI Portal", layout="wide")
st.title("👁️ Smart Assist: Environmental Awareness Dashboard")
st.write("This portal uses WebRTC to stream your camera frames to our AI server for live hazard detection.")

# --- Model Initialization ---
@st.cache_resource
def load_model():
    # Loads model once and caches it in server memory
    return YOLO("yolov8n.pt")

model = load_model()

# User defined focus hazards
FOCUS_CLASSES = [
    'person', 'stairs', 'pothole', 'door', 'table', 
    'vehicle', 'plant', 'fence', 'garbage bin'
]

# --- Sidebar Configuration ---
st.sidebar.header("System Calibration")
conf_val = st.sidebar.slider("Detection Confidence", 0.1, 1.0, 0.25)

# Storing threshold in session state so the video callback can read it dynamically
st.session_state["conf_threshold"] = conf_val

# WebRTC ICE servers configuration (ensures connection through strict firewalls)
RTC_CONFIGURATION = RTCConfiguration(
    {"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}
)

# --- Video Processing Engine ---
def video_frame_callback(frame: av.VideoFrame) -> av.VideoFrame:
    # 1. Convert incoming Web interface frame to a standard OpenCV BGR image
    img = frame.to_ndarray(format="bgr24")
    
    # 2. Advanced Low Light Logic (CLAHE)
    brightness = np.mean(img)
    if brightness < 80:
        lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        l = clahe.apply(l)
        img = cv2.cvtColor(cv2.merge((l, a, b)), cv2.COLOR_LAB2BGR)

    # 3. Read current slider value safely
    current_conf = st.session_state.get("conf_threshold", 0.25)

    # 4. YOLO Object Inference
    results = model(img, conf=current_conf, verbose=False)[0]
    h, w = img.shape[:2]

    # 5. Filter & Box Annotations
    for r in results.boxes:
        c_id = int(r.cls[0])
        name = model.names[c_id].lower()
        
        if any(fc in name for fc in FOCUS_CLASSES):
            x1, y1, x2, y2 = r.xyxy[0]
            box_w, box_h = int(x2 - x1), int(y2 - y1)
            closeness = ((box_w * box_h) / (w * h)) * 100
            
            # Change box boundary to Crimson Red if object is danger-close
            color = (0, 0, 255) if closeness > 25 else (0, 255, 0) 
            
            # Draw overlay markings
            cv2.rectangle(img, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
            cv2.putText(img, f"{name} {closeness:.1f}%", (int(x1), int(y1) - 10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

    # 6. Pass processed frame back to user's screen interface
    return av.VideoFrame.from_ndarray(img, format="bgr24")

# --- Render Live Interface Streamer ---
webrtc_streamer(
    key="smart-assist-engine",
    mode=WebRtcMode.SENDRECV,
    rtc_configuration=RTC_CONFIGURATION,
    video_frame_callback=video_frame_callback,
    media_stream_constraints={"video": True, "audio": False},
    async_processing=True,
)
