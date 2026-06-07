import streamlit as st
import cv2
import numpy as np
from ultralytics import YOLO
from gtts import gTTS
import av
import time
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase

# Set up widescreen configuration matching your original user interface template
st.set_page_config(
    page_title="Smart Assist Dashboard",
    page_icon="👁️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# -------------------------------------------------------------------------
# SIDEBAR: System Calibration (Restored from your dashboard UI layout)
# -------------------------------------------------------------------------
st.sidebar.title("🔧 System Calibration")

# Sliders calibrated to match your precise user parameters
confidence_threshold = st.sidebar.slider(
    "Detection Confidence", 
    min_value=0.0, max_value=1.0, value=0.25, step=0.05
)

st.sidebar.markdown("---")
st.sidebar.subheader("💡 Viva Presentation Tip")
st.sidebar.markdown("""
* **CLAHE Enhancement:** Applied dynamically below a mean brightness of 80 to ensure spatial clarity in low-light environments.
* **Proximity Metric:** Bounding box pixel density percentage relative to the canvas calculation determines hazard severity.
""")

# -------------------------------------------------------------------------
# RESOURCE CACHING: Optimized YOLO Model Load
# -------------------------------------------------------------------------
@st.cache_resource
def load_models():
    try:
        # Load standard lightweight models to run efficiently on low-compute containers
        return YOLO("yolov8n.pt")
    except Exception as e:
        st.sidebar.error(f"Error loading models: {e}")
        return None

model_gen = load_models()

# -------------------------------------------------------------------------
# AUDIO INTERFACE: Browser Audio Output Module
# -------------------------------------------------------------------------
def speak_text_in_browser(text_payload):
    """Generates an MP3 token and fires it directly via browser speakers."""
    if text_payload:
        try:
            tts = gTTS(text=text_payload, lang='en', tld='com')
            audio_path = "temp_alert.mp3"
            tts.save(audio_path)
            st.audio(audio_path, format="audio/mp3", autoplay=True)
        except Exception as e:
            pass

# -------------------------------------------------------------------------
# MAIN INTERFACE: Dynamic Layout Design
# -------------------------------------------------------------------------
st.title("👁️ Smart Assist: Environmental Awareness Portal")
st.caption("Development Stage: Local Hardware Prototype Interface")

# Main central application checkbox toggle from your original template screenshot
launch_engine = st.checkbox("Launch Smart Assist Webcam Engine", value=False)

if launch_engine:
    st.info("🎥 Stream active. Please ensure you allow your web browser camera permissions.")
    
    # Setup interface split columns for video feed vs text logs
    col1, col2 = col1, col2 = st.columns([2, 1])
    
    with col2:
        st.subheader("🔊 Audio Alerts Queue")
        status_box = st.empty()
        status_box.success("System Live: Scanning pathways...")
        metrics_placeholder = st.empty()
        alert_placeholder = st.empty()

    # Define the video callback engine that operates safely in asynchronous cloud threads
    class VideoProcessor(VideoProcessorBase):
        def __init__(self):
            self.low_light_announced = False
            self.last_alert_time = 0
            self.last_blur_time = 0

        def recv(self, frame):
            img = frame.to_ndarray(format="bgr24")
            img_h, img_w = img.shape[:2]
            current_time = time.time()
            
            # --- 1. Camera Focus/Sharpness Assessment ---
            focus_measure = cv2.Laplacian(img, cv2.CV_64F).var()
            camera_is_blur = focus_measure < 60.0
            
            # --- 2. Brightness Assessment & CLAHE ---
            avg_color = np.mean(img, axis=(0, 1))
            brightness = np.mean(avg_color)
            is_dark = brightness < 80
            
            processing_image = img.copy()
            
            if is_dark:
                lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
                l, a, b = cv2.split(lab)
                clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
                l = clahe.apply(l)
                processing_image = cv2.cvtColor(cv2.merge((l, a, b)), cv2.COLOR_LAB2BGR)
            
            # --- 3. Object Detection Inference Pipeline ---
            alert_parts = []
            trigger_beep = False
            
            if model_gen is not None:
                results = model_gen(processing_image, conf=confidence_threshold, verbose=False)[0]
                
                for box in results.boxes:
                    x1, y1, x2, y2 = box.xyxy[0].tolist()
                    class_id = int(box.cls[0])
                    class_name = model_gen.names[class_id]
                    confidence = float(box.conf[0])
                    
                    # --- 4. Distance Calculation via Box Area Metric ---
                    norm_area = ((x2 - x1) * (y2 - y1)) / (img_w * img_h)
                    dist_label = "far"
                    box_color = (0, 255, 0) # Green for normal tracking
                    
                    if norm_area > 0.25:
                        dist_label = "very close"
                        trigger_beep = True
                        box_color = (0, 0, 255) # Red bounding frames for close objects
                    elif norm_area > 0.08:
                        dist_label = "near"
                        box_color = (0, 165, 255) # Orange for medium indicators
                    
                    # --- 5. Direction Identification Calculus ---
                    center_x = (x1 + x2) / 2 / img_w
                    if center_x < 0.35:
                        dir_label = "to the left"
                    elif center_x > 0.65:
                        dir_label = "to the right"
                    else:
                        dir_label = "in front"
                    
                    alert_parts.append(f"{class_name} {dist_label} {dir_label}")
                    
                    # --- 6. Visualization Overlays ---
                    cv2.rectangle(processing_image, (int(x1), int(y1)), (int(x2), int(y2)), box_color, 2)
                    cv2.putText(processing_image, f"{class_name} {dist_label}", (int(x1), int(y1) - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, box_color, 2)

            # Send spatial state parameters to the display log variables
            metrics_placeholder.markdown(f"""
            **Diagnostics Logs:**
            * Sharpness Metric: `{focus_measure:.2f}` (Blur Limit: 60.0)
            * Ambient Luminosity: `{brightness:.2f}`
            """)
            
            if len(alert_parts) > 0:
                alert_placeholder.warning(f"Detected: {', '.join(alert_parts)}")
            else:
                alert_placeholder.info("Scanning... Pathway clear.")

            return av.VideoFrame.from_ndarray(processing_image, format="bgr24")

    with col1:
        # Deploy cloud-safe real-time WebRTC media streamer
        ctx = webrtc_streamer(
            key="smart-assist-streamer",
            video_processor_factory=VideoProcessor,
            rtc_configuration={"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]},
            media_stream_constraints={"video": True, "audio": False}
        )

else:
    # Restored your precise original layout instruction banner text from your image
    st.markdown("""
    <div style="background-color:#1e293b; padding:20px; border-radius:8px; border-left: 5px solid #3b82f6;">
        <span style="color:#94a3b8;">System Standby. Toggle the engine checkbox to boot up the vision tracking matrix.</span>
    </div>
    """, unsafe_allowed_html=True)
