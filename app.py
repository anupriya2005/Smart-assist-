import streamlit as st
from streamlit_webrtc import webrtc_streamer, WebRtcMode, RTCConfiguration
import av
import cv2
import numpy as np
from ultralytics import YOLO
import streamlit.components.v1 as components

# --- Page Setup ---
st.set_page_config(page_title="Smart Assist AI Portal", layout="wide")
st.title("👁️ Smart Assist: Environmental Awareness Dashboard")

# --- Model Initialization ---
@st.cache_resource
def load_model():
    return YOLO("yolov8n.pt")

model = load_model()

FOCUS_CLASSES = [
    'person', 'stairs', 'pothole', 'door', 'table', 
    'vehicle', 'plant', 'fence', 'garbage bin'
]

# --- Sidebar Configuration ---
st.sidebar.header("System Calibration")
conf_val = st.sidebar.slider("Detection Confidence", 0.1, 1.0, 0.25)
st.session_state["conf_threshold"] = conf_val

RTC_CONFIGURATION = RTCConfiguration(
    {"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}
)

# Initialize a session state variable to keep track of the last spoken alert
if "audio_queue" not in st.session_state:
    st.session_state["audio_queue"] = ""

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

    alert_triggered = False

    for r in results.boxes:
        c_id = int(r.cls[0])
        name = model.names[c_id].lower()
        
        if any(fc in name for fc in FOCUS_CLASSES):
            x1, y1, x2, y2 = r.xyxy[0]
            box_w, box_h = int(x2 - x1), int(y2 - y1)
            closeness = ((box_w * box_h) / (w * h)) * 100
            
            # Danger Close Alert (Object takes up more than 25% of the frame)
            if closeness > 25:
                color = (0, 0, 255) # Crimson Red
                if not alert_triggered:
                    # Update global string for audio payload
                    st.session_state["audio_queue"] = f"Warning. {name} is very close."
                    alert_triggered = True
            else:
                color = (0, 255, 0) # Green
            
            cv2.rectangle(img, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
            cv2.putText(img, f"{name} {closeness:.1f}%", (int(x1), int(y1) - 10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

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

# --- Browser-Native Text-to-Speech Engine (JavaScript Injection) ---
if st.session_state["audio_queue"] != "":
    text_to_speak = st.session_state["audio_queue"]
    
    # Inject JavaScript that calls the browser's speech synthesis engine
    tts_html = f"""
    <script>
        var msg = new SpeechSynthesisUtterance('{text_to_speak}');
        window.speechSynthesis.speak(msg);
    </script>
    """
    # Execute the component invisibly in the layout background
    components.html(tts_html, height=0, width=0)
    
    # Show text log for judges
    st.warning(f"🔊 Audio Output: \"{text_to_speak}\"")
    
    # Reset queue so it doesn't repeat infinitely
    st.session_state["audio_queue"] = ""
