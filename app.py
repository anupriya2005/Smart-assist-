import os
import base64
import cv2
import numpy as np
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
from ultralytics import YOLO
import easyocr
from gtts import gTTS

app = Flask(__name__)
CORS(app)  # Enable Cross-Origin Resource Sharing for frontend communication

# Create a directory to temporarily store generated audio responses
AUDIO_DIR = os.path.join(os.path.dirname(__file__), 'static', 'audio')
os.makedirs(AUDIO_DIR, exist_ok=True)

# -------------------------------------------------------------
# Initialize AI Models
# -------------------------------------------------------------
print("Loading YOLOv8 model...")
model = YOLO('yolov8n.pt')  # Using the nano model for fast, real-time CPU performance

print("Loading EasyOCR Reader...")
reader = easyocr.Reader(['en'], gpu=False)  # Set gpu=True if you have a CUDA setup


# -------------------------------------------------------------
# Helper Functions
# -------------------------------------------------------------
def decode_image(base64_string):
    """Decodes a base64 string sent by the frontend camera into an OpenCV image."""
    encoded_data = base64_string.split(',')[1]
    nparr = np.frombuffer(base64.b64decode(encoded_data), np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    return img

def generate_tts(text, filename="response.mp3"):
    """Converts text instructions into an MP3 file."""
    tts = gTTS(text=text, lang='en')
    filepath = os.path.join(AUDIO_DIR, filename)
    tts.save(filepath)
    return f"/static/audio/{filename}"


# -------------------------------------------------------------
# API Endpoints
# -------------------------------------------------------------
@app.route('/')
def home():
    return "Smart Assist AI Backend Running."

@app.route('/static/audio/<path:filename>')
def serve_audio(filename):
    """Serves the generated speech MP3 files to the frontend."""
    return send_from_directory(AUDIO_DIR, filename)

@app.route('/api/detect-objects', methods=['POST'])
def detect_objects():
    """Analyzes a camera frame to identify common objects and obstacles."""
    try:
        data = request.get_json()
        if not data or 'image' not in data:
            return jsonify({"error": "Missing image data"}), 400
        
        img = decode_image(data['image'])
        results = model(img, verbose=False)
        
        # Parse names of detected unique items
        detected_items = []
        for r in results:
            for c in r.boxes.cls:
                detected_items.append(model.names[int(c)])
        
        if detected_items:
            # Group identical items (e.g., "two chairs, one person")
            unique_items = set(detected_items)
            summary_list = [f"{detected_items.count(item)} {item}{'s' if detected_items.count(item) > 1 else ''}" for item in unique_items]
            response_text = "In front of you, I see: " + ", ".join(summary_list) + "."
        else:
            response_text = "The path ahead looks clear."
            
        audio_url = generate_tts(response_text, "detect.mp3")
        
        return jsonify({
            "text": response_text,
            "audio_url": audio_url
        })
        
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/api/read-text', methods=['POST'])
def read_text():
    """Extracts text signs, labels, or document words via OCR."""
    try:
        data = request.get_json()
        if not data or 'image' not in data:
            return jsonify({"error": "Missing image data"}), 400
            
        img = decode_image(data['image'])
        
        # Perform OCR text extraction
        ocr_result = reader.readtext(img, detail=0)
        
        if ocr_result:
            extracted_text = " ".join(ocr_result)
            response_text = f"The text reads: {extracted_text}"
        else:
            response_text = "No clear text could be detected in frame."
            
        audio_url = generate_tts(response_text, "ocr.mp3")
        
        return jsonify({
            "text": response_text,
            "audio_url": audio_url
        })
        
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/api/sos', methods=['POST'])
def trigger_sos():
    """Triggers mock emergency protocol and vocalizes confirmation."""
    # Note: In a production environment, you would call the Twilio or WhatsApp API here.
    emergency_msg = "Emergency protocol activated. Your location coordinates have been transmitted to your guardian."
    audio_url = generate_tts(emergency_msg, "sos.mp3")
    
    return jsonify({
        "text": emergency_msg,
        "audio_url": audio_url
    })

if __name__ == '__main__':
    # Flask application runs on port 5000
    app.run(host='0.0.0.0', port=5000, debug=True)
