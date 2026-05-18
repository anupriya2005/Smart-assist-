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
st.write("👉 *Click anywhere on the webpage or adjust the slider to activate browser audio permissions before starting the camera!*")

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

# A visible text box placeholder for live status updates
alert_placeholder = st.empty()

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
            
            # Keep track of the absolute closest object taking over the frame
            if closeness > 25 and closeness > max_closeness:
                max_closeness = closeness
                closest_object = name
                color = (0, 0, 255) # Red for danger-close
            else:
                color = (0, 255, 0) # Green
            
            cv2.rectangle(img, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
            cv2.putText(img, f"{name} {closeness:.1f}%", (int(x1), int(y1) - 10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

    # If an object is too close, dynamically inject browser-side speech via session attributes
    if closest_object:
        st.session_state["text_to_speak"] = f"Warning! {closest_object} is very close."
    else:
        st.session_state["text_to_speak"] = ""

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

# --- Browser Text-to-Speech Engine via JavaScript ---
# Checking text state every frontend UI render cycle
speech_text = st.session_state.get("text_to_speak", "")

if speech_text:
    # Display warning layout bar
    alert_placeholder.error(f"🔊 AI Audio Output: \"{speech_text}\"")
    
    # JavaScript logic with safety timeout to avoid freezing your dashboard window
    tts_javascript = f"""
    <script>
        if ('speechSynthesis' in window) {{
            // Cancel any ongoing speech so it doesn't backlog
            window.speechSynthesis.cancel(); 
            var speech = new SpeechSynthesisUtterance("{speech_text}");
            speech.lang = 'en-US';
            speech.rate = 1.0;
            window.speechSynthesis.speak(speech);
        }}
    </script>
    """
    # Execute Javascript element invisibly in browser background
    components.html(tts_javascript, height=0)
else:
    alert_placeholder.success("🟢 System Clear: No hazards immediate.")
# Add this at the absolute bottom of app.py
st.markdown("---")
st.subheader("🔊 Audio Helper")

# If an object was detected, create a button to force the browser to speak
if st.session_state.get("text_to_speak", ""):
    alert_msg = st.session_state["text_to_speak"]
    
    if st.button("📢 Click to Hear Audio Alert"):
        tts_button_js = f"""
        <script>
            window.speechSynthesis.cancel();
            var speech = new SpeechSynthesisUtterance("{alert_msg}");
            window.speechSynthesis.speak(speech);
        </script>
        """
        components.html(tts_button_js, height=0)
