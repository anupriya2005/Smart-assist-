import streamlit as st
from streamlit_webrtc import webrtc_streamer, WebRtcMode, RTCConfiguration
import av
import cv2
import numpy as np
from ultralytics import YOLO
import streamlit.components.v1 as components
import os

# --- Page Setup ---
st.set_page_config(page_title="Smart Assist AI Portal", layout="wide")
st.title("👁️ Smart Assist: Environmental Awareness Dashboard")
st.write("👉 *Click anywhere on this text or adjust the slider to grant browser audio permissions before starting the camera!*")

# --- Model Initialization ---
@st.cache_resource
def load_model():
    return YOLO("yolov8n.pt")

model = load_model()

FOCUS_CLASSES = [
    'person', 'stairs', 'pothole', 'door', 'table', 
    'vehicle', 'plant', 'fence', 'garbage bin'
]

# Temporary file path to pass alert data safely across threads
ALERT_FILE = "latest_alert.txt"

# --- Sidebar Configuration ---
st.sidebar.header("System Calibration")
conf_val = st.sidebar.slider("Detection Confidence", 0.1, 1.0, 0.25)
st.session_state["conf_threshold"] = conf_val

RTC_CONFIGURATION = RTCConfiguration(
    {"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}
)

# --- Video Processing Engine (Runs on Background Thread) ---
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

    # Read current slider value safely from session state
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
                color = (0, 0, 255) # Red for danger-close
            else:
                color = (0, 255, 0) # Green
            
            cv2.rectangle(img, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
            cv2.putText(img, f"{name} {closeness:.1f}%", (int(x1), int(y1) - 10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

    # Write alert data to a file instead of updating session_state directly
    if closest_object:
        with open(ALERT_FILE, "w") as f:
            f.write(f"Warning! {closest_object} is very close.")
    else:
        if os.path.exists(ALERT_FILE):
            try:
                os.remove(ALERT_FILE)
            except:
                pass

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

# --- Frontend Audio/UI Sync (Runs on Main Thread) ---
# Read the file to see if the video thread flagged an alert
speech_text = ""
if os.path.exists(ALERT_FILE):
    try:
        with open(ALERT_FILE, "r") as f:
            speech_text = f.read()
    except:
        pass

# Display UI updates and trigger browser JavaScript audio safely
if speech_text:
    st.error(f"🔊 AI Audio Output Log: \"{speech_text}\"")
    
    # JavaScript logic to execute text-to-speech right in the browser window
    tts_javascript = f"""
    <script>
        if ('speechSynthesis' in window) {{
            window.speechSynthesis.cancel(); // Prevent audio overlap queue
            var speech = new SpeechSynthesisUtterance("{speech_text}");
            speech.lang = 'en-US';
            speech.rate = 1.1;
            window.speechSynthesis.speak(speech);
        }}
    </script>
    """
    components.html(tts_javascript, height=0)
    
    # Backup Click-to-Hear Button in case browser blocks the autoplay audio
    if st.button("📢 Click here if you cannot hear the automatic voice alert"):
        components.html(tts_javascript, height=0)
else:
    st.success("🟢 System Clear: No hazards immediate.")
