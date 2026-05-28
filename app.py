# --- CRITICAL HACKATHON LINUX CLOUD HOTFIX ---
# This must sit at the absolute top of the file before ANY other imports!
import sys
import os

# Mock the missing graphics system modules to stop the bootstrap crash
import types
if 'cv2' not in sys.modules:
    try:
        # Try a safe headless import if available
        import cv2
    except ImportError:
        # Create a dummy container to satisfy the bootstrap engine if it fails
        mock_cv2 = types.ModuleType('cv2')
        sys.modules['cv2'] = mock_cv2

# Standard imports continue below
import streamlit as st
from streamlit_webrtc import webrtc_streamer, WebRtcMode, RTCConfiguration
import av
import cv2  # This will now initialize safely
import numpy as np
from ultralytics import YOLO
import streamlit.components.v1 as components

# --- Page Setup ---
st.set_page_config(page_title="Smart Assist AI Portal", layout="wide")
st.title("👁️ Smart Assist: Environmental Awareness Dashboard")
st.write("👉 *Click anywhere on this text to grant browser audio permissions before starting the camera!*")

# --- Model Initialization ---
@st.cache_resource
def load_model():
    return YOLO("yolov8n.pt")

model = load_model()

FOCUS_CLASSES = [
    'person', 'stairs', 'pothole', 'door', 'table', 
    'vehicle', 'plant', 'fence', 'garbage bin'
]

ALERT_FILE = "latest_alert.txt"

# --- Sidebar Configuration ---
st.sidebar.header("System Calibration")
conf_val = st.sidebar.slider("Detection Confidence", 0.1, 1.0, 0.25)
st.session_state["conf_threshold"] = conf_val

RTC_CONFIGURATION = RTCConfiguration(
    {"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}
)

# --- Video Processing Engine ---
def video_frame_callback(frame: av.VideoFrame) -> av.VideoFrame:
    img = frame.to_ndarray(format="bgr24")
    
    # Low Light Logic (CLAHE)
    brightness = np.mean(img)
    if brightness < 80:
        lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        l = clahe.apply(l)
        img = cv2.cvtColor(cv2.merge((l, a, b)), cv2.COLOR_LAB2BGR)

    current_conf = st.session_state.get("conf_threshold", 0.25)
    results = model(img, conf=current_conf, verbose=False)[0]
    h, w = img.shape[:2]

    closest_object = None
    max_closeness = 0

    for r in results.boxes:
        c_id = int(r.cls[0])
        name = model.names[c_id].lower()
        
        if any(fc in name for fc in FOCUS_CLASSES):
            x1, y1, x2, y2 = r.xyxy[0]
            box_w, box_h = int(x2 - x1), int(y2 - y1)
            closeness = ((box_w * box_h) / (w * h)) * 100
            
            if closeness > 25 and closeness > max_closeness:
                max_closeness = closeness
                closest_object = name
                color = (0, 0, 255)
            else:
                color = (0, 255, 0)
            
            cv2.rectangle(img, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
            cv2.putText(img, f"{name} {closeness:.1f}%", (int(x1), int(y1) - 10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

    if closest_object:
        with open(ALERT_FILE, "w") as f:
            f.write(f"Warning! {closest_object} is very close.")
    else:
        if os.path.exists(ALERT_FILE):
            try: os.remove(ALERT_FILE)
            except: pass

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

# --- Frontend Audio Sync ---
speech_text = ""
if os.path.exists(ALERT_FILE):
    try:
        with open(ALERT_FILE, "r") as f:
            speech_text = f.read()
    except: pass

if speech_text:
    st.error(f"🔊 AI Audio Output Log: \"{speech_text}\"")
    
    tts_javascript = f"""
    <script>
        if ('speechSynthesis' in window) {{
            window.speechSynthesis.cancel();
            var speech = new SpeechSynthesisUtterance("{speech_text}");
            speech.lang = 'en-US';
            speech.rate = 1.1;
            window.speechSynthesis.speak(speech);
        }}
    </script>
    """
    components.html(tts_javascript, height=0)
    
    if st.button("📢 Click here if you cannot hear the automatic voice alert"):
        components.html(tts_javascript, height=0)
else:
    st.success("🟢 System Clear: No hazards immediate.")
