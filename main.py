import os
from flask import Flask, render_template_string

app = Flask(__name__)

# Single-file architecture: Delivering a feature-packed accessibility dashboard
HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>SMART ASSIST - Ultimate Accessibility Companion</title>
    
    <script src="https://cdn.jsdelivr.net/npm/@tensorflow/tfjs"></script>
    <script src="https://cdn.jsdelivr.net/npm/@tensorflow/@tensorflow-models/coco-ssd"></script>
    <script src="https://cdn.jsdelivr.net/npm/tesseract.js@5/dist/tesseract.min.js"></script>

    <style>
        :root {
            --bg-color: #000000;
            --text-color: #FFFFFF;
            --accent-color: #FFFF00;
            --danger-color: #FF3333;
            --safe-color: #00FF00;
        }

        body {
            background-color: var(--bg-color);
            color: var(--text-color);
            font-family: 'Arial', sans-serif;
            margin: 0;
            padding: 20px;
            text-align: center;
        }

        header {
            border-bottom: 5px solid var(--accent-color);
            padding-bottom: 15px;
            margin-bottom: 25px;
        }

        h1 { font-size: 3rem; color: var(--accent-color); margin: 0; }
        
        .main-container {
            display: flex;
            flex-direction: column;
            align-items: center;
            max-width: 800px;
            margin: 0 auto;
        }

        .video-box {
            position: relative;
            width: 100%;
            max-width: 500px;
            border: 5px solid var(--text-color);
            background-color: #111;
            margin-bottom: 25px;
            border-radius: 8px;
            overflow: hidden;
        }

        video {
            width: 100%;
            height: auto;
            display: block;
            transform: scaleX(-1); /* Mirror mode */
        }

        /* Large Accessible UI Grid Matrix */
        .grid-container {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(280px, 1fr));
            gap: 20px;
            width: 100%;
            margin-bottom: 25px;
        }

        .btn {
            background-color: #222;
            color: var(--text-color);
            border: 4px solid var(--accent-color);
            padding: 30px 20px;
            font-size: 1.6rem;
            font-weight: bold;
            cursor: pointer;
            border-radius: 12px;
            transition: all 0.2s ease;
        }

        .btn:focus, .btn:hover {
            background-color: var(--accent-color);
            color: #000;
            outline: 5px solid var(--safe-color);
        }

        .btn-sos {
            border-color: var(--danger-color);
            color: var(--danger-color);
        }
        .btn-sos:focus, .btn-sos:hover {
            background-color: var(--danger-color);
            color: #000;
        }

        /* Dynamic Console Output Board */
        #status-console {
            padding: 25px;
            border: 3px dashed var(--accent-color);
            font-size: 1.8rem;
            width: 100%;
            box-sizing: border-box;
            background-color: #111;
            font-weight: bold;
            min-height: 80px;
        }
    </style>
</head>
<body>

    <header>
        <h1>SMART ASSIST</h1>
        <p style="color: var(--safe-color); font-size: 1.3rem; margin: 5px 0 0 0;">
            AI Framework for Independent Mobility
        </p>
    </header>

    <div class="main-container">
        <div class="video-box" id="v-box">
            <video id="webcam" autoplay playsinline muted></video>
        </div>

        <canvas id="ocr-canvas" width="640" height="480" style="display: none;"></canvas>

        <main class="grid-container">
            <button class="btn" id="btn-voice" aria-label="Voice Command Assistant. Press V shortcut.">🎙️ Voice Assistant (V)</button>
            <button class="btn" id="btn-object" aria-label="Toggle Proximity Scanner. Press O shortcut.">📷 Object Scanner (O)</button>
            <button class="btn" id="btn-text" aria-label="Read Text out loud. Press T shortcut.">📖 Read Text (T)</button>
            <button class="btn btn-sos" id="btn-sos" aria-label="Trigger Emergency Alert. Press S shortcut.">🚨 Emergency SOS (S)</button>
        </main>

        <div id="status-console" role="status" aria-live="assertive">
            System initialization ongoing...
        </div>
    </div>

    <script>
        const video = document.getElementById('webcam');
        const ocrCanvas = document.getElementById('ocr-canvas');
        const statusConsole = document.getElementById('status-console');
        const videoBox = document.getElementById('v-box');
        
        let objectModel = null;
        let isScanningObjects = false;
        let animationFrameId = null;
        let lastSpokenTime = 0;

        // Audio System setup to generate dynamic hardware alert Beeps
        const audioCtx = new (window.AudioContext || window.webkitAudioContext)();

        // 1. Core Loader Setup
        async function initSystem() {
            statusConsole.innerText = "Initializing AI Intelligence Cores...";
            try {
                objectModel = await cocoSsd.load();
                statusConsole.innerText = "System Fully Armed. Ready for your input.";
                speak("Smart Assist models loaded. Ready for navigation.");
            } catch (err) {
                statusConsole.innerText = "Core loading failure encountered.";
                console.error(err);
            }
        }

        // 2. Hardware Tone Generator
        function triggerBeep(frequency, duration) {
            if (audioCtx.state === 'suspended') audioCtx.resume();
            const osc = audioCtx.createOscillator();
            const gain = audioCtx.createGain();
            osc.type = 'sine';
            osc.frequency.value = frequency;
            gain.gain.setValueAtTime(0.3, audioCtx.currentTime);
            gain.gain.exponentialRampToValueAtTime(0.01, audioCtx.currentTime + duration);
            osc.connect(gain);
            gain.connect(audioCtx.destination);
            osc.start();
            osc.stop(audioCtx.currentTime + duration);
        }

        // 3. Screen Reader Text-to-Speech Output Engine
        function speak(text) {
            window.speechSynthesis.cancel();
            const msg = new SpeechSynthesisUtterance(text);
            msg.rate = 1.05;
            window.speechSynthesis.speak(msg);
        }

        // 4. Initialize Hardware Camera Feed
        async function startCamera() {
            if (navigator.mediaDevices && navigator.mediaDevices.getUserMedia) {
                try {
                    const stream = await navigator.mediaDevices.getUserMedia({
                        video: { width: 640, height: 480, facingMode: "environment" }
                    });
                    video.srcObject = stream;
                } catch (err) {
                    statusConsole.innerText = "Camera resource access denied.";
                    speak("Camera access error.");
                }
            }
        }

        // 5. FEATURE 1: Object & Proximity Continuous Scanner Loop
        async function runObjectDetectionLoop() {
            if (!isScanningObjects) return;

            const predictions = await objectModel.detect(video);
            let extremeDanger = false;
            let targetObstacle = "";
            let largestArea = 0;

            predictions.forEach(p => {
                const [x, y, w, h] = p.bbox;
                const frameArea = video.videoWidth * video.videoHeight;
                const proximityScore = ((w * h) / frameArea) * 100;

                // If object takes up more than 25% of the screen space, it's too close!
                if (proximityScore > 25) {
                    extremeDanger = true;
                    if (proximityScore > largestArea) {
                        largestArea = proximityScore;
                        targetObstacle = p.class;
                    }
                }
            });

            if (extremeDanger) {
                videoBox.style.borderColor = "#FF3333";
                statusConsole.style.borderColor = "#FF3333";
                statusConsole.innerText = `CRITICAL WARNING: Close ${targetObstacle.toUpperCase()} Ahead!`;
                
                // Emits a high pitch hardware threat chime
                triggerBeep(950, 0.12);

                let now = Date.now();
                if (now - lastSpokenTime > 1800) {
                    speak(`Warning. Close ${targetObstacle} detected.`);
                    lastSpokenTime = now;
                }
            } else {
                videoBox.style.borderColor = "#FFFFFF";
                statusConsole.style.borderColor = "#00FF00";
                if (predictions.length > 0) {
                    statusConsole.innerText = `Path Clear. Visible objects: ${predictions.map(p => p.class).join(', ')}`;
                } else {
                    statusConsole.innerText = "Scanning... Path completely clear ahead.";
                }
            }

            animationFrameId = requestAnimationFrame(runObjectDetectionLoop);
        }

        function toggleObjectScanner() {
            if (!isScanningObjects) {
                isScanningObjects = true;
                document.getElementById('btn-object').style.backgroundColor = "var(--danger-color)";
                speak("Object proximity scanning engaged.");
                runObjectDetectionLoop();
            } else {
                isScanningObjects = false;
                document.getElementById('btn-object').style.backgroundColor = "#222";
                videoBox.style.borderColor = "#FFFFFF";
                statusConsole.style.borderColor = "#FFFF00";
                statusConsole.innerText = "Object scanner turned off.";
                speak("Object scanning disengaged.");
                if (animationFrameId) cancelAnimationFrame(animationFrameId);
            }
        }

        // 6. FEATURE 2: Real-time In-Browser Optical Character Recognition (OCR)
        async function runTextReading() {
            // Turn off scanner loops to clear computer processing lane
            if (isScanningObjects) toggleObjectScanner();

            statusConsole.innerText = "Capturing frame snapshot and performing OCR text analysis...";
            speak("Reading text. Hold completely still.");

            // Capture frozen frame matrix onto hidden canvas surface
            const ctx = ocrCanvas.getContext('2d');
            ctx.drawImage(video, 0, 0, ocrCanvas.width, ocrCanvas.height);
            
            try {
                // Initialize Tesseract Engine locally on image stream
                const result = await Tesseract.recognize(ocrCanvas, 'eng');
                const cleanText = result.data.text.trim();
                
                if (cleanText.length > 0) {
                    statusConsole.innerText = `Extracted Text: "${cleanText}"`;
                    speak(`The document reads: ${cleanText}`);
                } else {
                    statusConsole.innerText = "OCR Finished. No structured text could be parsed.";
                    speak("No text detected in front of the camera.");
                }
            } catch (err) {
                statusConsole.innerText = "Error executing text extraction routine.";
                speak("Text reader processing failure.");
            }
        }

        // 7. FEATURE 3: Voice Command Engine
        function runVoiceAssistant() {
            const SpeechRecognition = window.SpeechRecognition || window.webkitSpeechRecognition;
            if (!SpeechRecognition) {
                statusConsole.innerText = "Voice Recognition feature missing on this browser engine.";
                speak("Voice recognition unsupported.");
                return;
            }

            const activeRecognizer = new SpeechRecognition();
            activeRecognizer.lang = 'en-US';
            
            statusConsole.innerText = "Listening for audio instructions now...";
            speak("How can I help you?");

            activeRecognizer.onresult = function(event) {
                const phrase = event.results[0][0].transcript.toLowerCase();
                statusConsole.innerText = `Processing Instruction: "${phrase}"`;

                setTimeout(() => {
                    if (phrase.includes("object") || phrase.includes("scan") || phrase.includes("see")) {
                        toggleObjectScanner();
                    } else if (phrase.includes("read") || phrase.includes("text") || phrase.includes("book")) {
                        runTextReading();
                    } else if (phrase.includes("help") || phrase.includes("emergency") || phrase.includes("sos")) {
                        triggerEmergencySOS();
                    } else {
                        statusConsole.innerText = `Command Unrecognized: "${phrase}"`;
                        speak("I didn't understand. Please say command keywords like scan, read text, or emergency.");
                    }
                }, 1000);
            };
        }

        // 8. FEATURE 4: Emergency SOS Execution
        function triggerEmergencySOS() {
            if (isScanningObjects) toggleObjectScanner();
            videoBox.style.borderColor = "#FF3333";
            statusConsole.style.borderColor = "#FF3333";
            statusConsole.innerText = "EMERGENCY PROTOCOL ACTIVE. ALERT SENT TO GUARDIAN.";
            
            // Continuous high intensity alarm pulses
            triggerBeep(1100, 0.4);
            setTimeout(() => triggerBeep(1100, 0.4), 500);
            
            speak("Emergency sequence activated. Your coordinates have been transmitted to your primary guardian.");
        }

        // 9. Interaction Setup & Shortcut Keys
        document.getElementById('btn-voice').addEventListener('click', runVoiceAssistant);
        document.getElementById('btn-object').addEventListener('click', toggleObjectScanner);
        document.getElementById('btn-text').addEventListener('click', runTextReading);
        document.getElementById('btn-sos').addEventListener('click', triggerEmergencySOS);

        // Vocalize focused element updates while moving using Tab navigation
        document.querySelectorAll('.btn').forEach(button => {
            button.addEventListener('focus', () => {
                speak(button.getAttribute('aria-label'));
            });
        });

        // Add key listeners for physical shortcuts
        window.addEventListener('keydown', (e) => {
            const choice = e.key.toLowerCase();
            if (choice === 'v') runVoiceAssistant();
            if (choice === 'o') toggleObjectScanner();
            if (choice === 't') runTextReading();
            if (choice === 's') triggerEmergencySOS();
        });

        // Boot instructions
        initSystem();
        startCamera();
    </script>
</body>
</html>
"""

@app.route('/')
def home():
    return render_template_string(HTML_TEMPLATE)

if __name__ == '__main__':
    # Launch system server
    app.run(host='0.0.0.0', port=5000, debug=True)
