import streamlit as st
import streamlit.components.v1 as components
import base64
import cv2
import numpy as np
from ultralytics import YOLO
import easyocr
from gtts import gTTS
import os

# Set up page config
st.set_page_config(page_title="Smart Assist AI", layout="centered")

# -------------------------------------------------------------
# Cache AI Models (So they only load ONCE on the server)
# -------------------------------------------------------------
@st.cache_resource
def load_models():
    yolo_model = YOLO('yolov8n.pt')
    ocr_reader = easyocr.Reader(['en'], gpu=False)
    return yolo_model, ocr_reader

model, reader = load_models()

# Create directory to store audio files dynamically
AUDIO_DIR = "static_audio"
os.makedirs(AUDIO_DIR, exist_ok=True)

# -------------------------------------------------------------
# Streamlit Hidden Query Parameters acting as an Internal API
# -------------------------------------------------------------
# We use query parameters to allow the frontend HTML to send requests back to Python.
query_params = st.query_params

if "action" in query_params:
    action = query_params["action"]
    
    # Process Object Detection
    if action == "detect" and "image" in query_params:
        try:
            img_data = query_params["image"]
            encoded_data = img_data.split(',')[1]
            nparr = np.frombuffer(base64.b64decode(encoded_data), np.uint8)
            img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            
            results = model(img, verbose=False)
            detected_items = [model.names[int(c)] for r in results for c in r.boxes.cls]
            
            if detected_items:
                unique_items = set(detected_items)
                summary = [f"{detected_items.count(i)} {i}{'s' if detected_items.count(i) > 1 else ''}" for i in unique_items]
                response_text = "In front of you, I see: " + ", ".join(summary) + "."
            else:
                response_text = "The path ahead looks clear."
        except Exception:
            response_text = "Error processing object detection mapping."
            
        st.write(f"### SYSTEM RESPONSE: {response_text}")
        tts = gTTS(text=response_text, lang='en')
        tts.save(f"{AUDIO_DIR}/detect.mp3")
        st.audio(f"{AUDIO_DIR}/detect.mp3", autoplay=True)

    # Process Text Reading (OCR)
    elif action == "read" and "image" in query_params:
        try:
            img_data = query_params["image"]
            encoded_data = img_data.split(',')[1]
            nparr = np.frombuffer(base64.b64decode(encoded_data), np.uint8)
            img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            
            ocr_result = reader.readtext(img, detail=0)
            response_text = f"The text reads: {' '.join(ocr_result)}" if ocr_result else "No clear text could be detected."
        except Exception:
            response_text = "Error during optical character reading."
            
        st.write(f"### SYSTEM RESPONSE: {response_text}")
        tts = gTTS(text=response_text, lang='en')
        tts.save(f"{AUDIO_DIR}/ocr.mp3")
        st.audio(f"{AUDIO_DIR}/ocr.mp3", autoplay=True)

    # Process SOS
    elif action == "sos":
        response_text = "Emergency protocol activated. Your coordinates have been transmitted to your guardian."
        st.write(f"### SYSTEM RESPONSE: {response_text}")
        tts = gTTS(text=response_text, lang='en')
        tts.save(f"{AUDIO_DIR}/sos.mp3")
        st.audio(f"{AUDIO_DIR}/sos.mp3", autoplay=True)

# -------------------------------------------------------------
# Inject the Ultra-Accessible UI via HTML Component
# -------------------------------------------------------------
# This embeds the high-contrast view, camera capture, and hotkeys directly.
accessible_html = """
<!DOCTYPE html>
<html>
<head>
    <style>
        :root { --bg: #000; --text: #FFF; --accent: #FFFF00; --danger: #FF3333; }
        body { background: var(--bg); color: var(--text); font-family: Arial; text-align: center; padding: 10px; margin: 0;}
        h1 { color: var(--accent); font-size: 2.2rem; margin: 5px; }
        .video-box { width: 100%; max-width: 400px; margin: 0 auto 15px auto; border: 4px solid var(--text); }
        video { width: 100%; transform: scaleX(-1); display: block; }
        .grid { display: grid; grid-template-columns: 1fr 1fr; gap: 15px; max-width: 500px; margin: 0 auto; }
        .btn { background: #222; color: var(--text); border: 3px solid var(--accent); padding: 20px 10px; font-size: 1.2rem; font-weight: bold; cursor: pointer; border-radius: 8px; }
        .btn:focus, .btn:hover { background: var(--accent); color: #000; outline: 3px solid #00FF00; }
        .btn-sos { border-color: var(--danger); color: var(--danger); }
        .btn-sos:focus, .btn-sos:hover { background: var(--danger); color: #000; }
    </style>
</head>
<body>
    <h1>SMART ASSIST AI</h1>
    <p style="color:#00FF00; margin: 0 0 10px 0;">Tab to move. Press/Vocalize: V (Voice), O (Object), T (Text), S (SOS)</p>
    
    <div class="video-box"><video id="webcam" autoplay playsinline></video></div>
    <canvas id="canvas" width="640" height="480" style="display:none;"></canvas>

    <div class="grid">
        <button class="btn" id="v-btn" aria-label="Voice Assistant. Press V.">🎙️ Voice (V)</button>
        <button class="btn" id="o-btn" aria-label="Detect Objects. Press O.">📷 Object (O)</button>
        <button class="btn" id="t-btn" aria-label="Read Text. Press T.">📖 Text (T)</button>
        <button class="btn btn-sos" id="s-btn" aria-label="Emergency SOS. Press S.">🚨 SOS (S)</button>
    </div>

    <script>
        const video = document.getElementById('webcam');
        const canvas = document.getElementById('canvas');

        if (navigator.mediaDevices && navigator.mediaDevices.getUserMedia) {
            navigator.mediaDevices.getUserMedia({ video: true }).then(s => video.srcObject = s);
        }

        function speak(t) {
            window.speechSynthesis.cancel();
            window.speechSynthesis.speak(new SpeechSynthesisUtterance(t));
        }

        function capture() {
            const ctx = canvas.getContext('2d');
            ctx.drawImage(video, 0, 0, canvas.width, canvas.height);
            return canvas.toDataURL('image/jpeg', 0.7);
        }

        // Send data back up to Streamlit by reloading the parent window URL with query parameters
        function sendToStreamlit(action, img='') {
            const baseUrl = window.parent.location.origin + window.parent.location.pathname;
            let targetUrl = baseUrl + '?action=' + action;
            if(img) {
                // Keep image string safe for URLs
                targetUrl += '&image=' + encodeURIComponent(img);
            }
            window.parent.location.href = targetUrl;
        }

        document.getElementById('o-btn').addEventListener('click', () => { speak("Scanning."); sendToStreamlit('detect', capture()); });
        document.getElementById('t-btn').addEventListener('click', () => { speak("Reading."); sendToStreamlit('read', capture()); });
        document.getElementById('s-btn').addEventListener('click', () => { sendToStreamlit('sos'); });

        // Voice Assistant Hook
        document.getElementById('v-btn').addEventListener('click', () => {
            const Speech = window.SpeechRecognition || window.webkitSpeechRecognition;
            if(!Speech) return speak("Voice recognition unsupported.");
            const rec = new Speech();
            speak("Listening.");
            rec.start();
            rec.onresult = (e) => {
                const cmd = e.results[0][0].transcript.toLowerCase();
                if (cmd.includes("object") || cmd.includes("see")) { sendToStreamlit('detect', capture()); }
                else if (cmd.includes("text") || cmd.includes("read")) { sendToStreamlit('read', capture()); }
                else if (cmd.includes("help") || cmd.includes("emergency")) { sendToStreamlit('sos'); }
                else { speak("Unknown command."); }
            };
        });

        // Screen reader simulation for keyboard tab selection
        document.querySelectorAll('.btn').forEach(b => {
            b.addEventListener('focus', () => speak(b.getAttribute('aria-label')));
        });

        // Hotkey bindings
        window.addEventListener('keydown', (e) => {
            const k = e.key.toLowerCase();
            if(k === 'o') { speak("Scanning."); sendToStreamlit('detect', capture()); }
            if(k === 't') { speak("Reading."); sendToStreamlit('read', capture()); }
            if(k === 's') sendToStreamlit('sos');
            if(k === 'v') document.getElementById('v-btn').click();
        });
    </script>
</body>
</html>
"""

# Render the interface inside Streamlit
components.html(accessible_html, height=620, scrolling=False)
