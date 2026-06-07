import streamlit as st
import streamlit.components.v1 as components

st.set_page_config(page_title="Smart Assist AI", layout="centered")

ACCESSIBLE_UI_HTML = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>SMART ASSIST - Ultimate Accessibility Companion</title>
    
    <script src="https://unpkg.com/@tensorflow/tfjs@4.22.0/dist/tf.min.js"></script>
    <script src="https://unpkg.com/@tensorflow-models/coco-ssd@2.2.3/dist/coco-ssd.min.js"></script>
    <script src="https://unpkg.com/tesseract.js@5.1.0/dist/tesseract.min.js"></script>
    
    <style>
        :root { --bg-color: #000000; --text-color: #FFFFFF; --accent-color: #FFFF00; --danger-color: #FF3333; --safe-color: #00FF00; }
        body { background-color: var(--bg-color); color: var(--text-color); font-family: 'Arial', sans-serif; margin: 0; padding: 10px; text-align: center; }
        header { border-bottom: 5px solid var(--accent-color); padding-bottom: 10px; margin-bottom: 20px; }
        h1 { font-size: 2.5rem; color: var(--accent-color); margin: 0; }
        .video-box { position: relative; width: 100%; max-width: 450px; border: 5px solid var(--text-color); background-color: #111; margin: 0 auto 20px auto; border-radius: 8px; overflow: hidden; }
        video { width: 100%; height: auto; display: block; transform: scaleX(-1); background: #222; min-height: 250px; }
        .grid-container { display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 15px; width: 100%; max-width: 600px; margin: 0 auto 20px auto; }
        .btn { background-color: #222; color: var(--text-color); border: 4px solid var(--accent-color); padding: 25px 15px; font-size: 1.4rem; font-weight: bold; cursor: pointer; border-radius: 12px; }
        .btn:focus, .btn:hover { background-color: var(--accent-color); color: #000; outline: 5px solid var(--safe-color); }
        .btn-sos { border-color: var(--danger-color); color: var(--danger-color); }
        .btn-sos:focus, .btn-sos:hover { background-color: var(--danger-color); color: #000; }
        #status-console { padding: 20px; border: 3px dashed var(--accent-color); font-size: 1.6rem; background-color: #111; font-weight: bold; min-height: 60px; max-width: 600px; margin: 0 auto; box-sizing: border-box; }
    </style>
</head>
<body>
    <header>
        <h1>SMART ASSIST</h1>
        <p style="color: var(--safe-color); font-size: 1.1rem; margin: 5px 0 0 0;">AI Framework for Independent Mobility</p>
    </header>
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
    <div id="status-console" role="status" aria-live="assertive">Initializing AI Intelligence Cores...</div>

    <script>
        const video = document.getElementById('webcam');
        const ocrCanvas = document.getElementById('ocr-canvas');
        const statusConsole = document.getElementById('status-console');
        const videoBox = document.getElementById('v-box');
        let objectModel = null; let isScanningObjects = false; let animationFrameId = null; let lastSpokenTime = 0;
        const audioCtx = new (window.AudioContext || window.webkitAudioContext)();

        async function initSystem() {
            try {
                // Fail-safe check for Global Model availability
                let checkInterval = setInterval(async () => {
                    if (typeof cocoSsd !== 'undefined' && typeof tf !== 'undefined') {
                        clearInterval(checkInterval);
                        objectModel = await cocoSsd.load();
                        statusConsole.innerText = "System Fully Armed. Ready for input.";
                        speak("Smart Assist models loaded.");
                    }
                }, 500);
                
                // Safety timeout if network fails entirely
                setTimeout(() => {
                    if(!objectModel) {
                        clearInterval(checkInterval);
                        statusConsole.style.borderColor = "#FF3333";
                        statusConsole.innerText = "CONNECTION TIMEOUT: Ensure you are using an HTTPS:// address.";
                    }
                }, 10000);

            } catch (err) { 
                statusConsole.style.borderColor = "#FF3333";
                statusConsole.innerText = "CRITICAL ERROR: AI Assets blocked."; 
            }
        }

        function triggerBeep(freq, dur) {
            try {
                if (audioCtx.state === 'suspended') audioCtx.resume();
                const osc = audioCtx.createOscillator(); const gain = audioCtx.createGain();
                osc.type = 'sine'; osc.frequency.value = freq;
                gain.gain.setValueAtTime(0.3, audioCtx.currentTime);
                gain.gain.exponentialRampToValueAtTime(0.01, audioCtx.currentTime + dur);
                osc.connect(gain); gain.connect(audioCtx.destination);
                osc.start(); osc.stop(audioCtx.currentTime + dur);
            } catch(e) {}
        }

        function speak(text) {
            try {
                window.speechSynthesis.cancel();
                const msg = new SpeechSynthesisUtterance(text); msg.rate = 1.05;
                window.speechSynthesis.speak(msg);
            } catch(e) {}
        }

        async function startCamera() {
            try {
                const stream = await navigator.mediaDevices.getUserMedia({ 
                    video: { width: 640, height: 480, facingMode: "environment" } 
                });
                video.srcObject = stream;
            } catch (err) { 
                statusConsole.style.borderColor = "#FF3333";
                statusConsole.innerText = "CAMERA BLOCKED: Please grant browser camera accessibility permission."; 
                speak("Camera blocked.");
            }
        }

        async function runObjectDetectionLoop() {
            if (!isScanningObjects || !objectModel) return;
            try {
                const predictions = await objectModel.detect(video);
                let extremeDanger = false; let targetObstacle = ""; let largestArea = 0;
                
                predictions.forEach(p => {
                    const [x, y, w, h] = p.bbox;
                    const score = ((w * h) / (video.videoWidth * video.videoHeight)) * 100;
                    if (score > 25) { extremeDanger = true; if (score > largestArea) { largestArea = score; targetObstacle = p.class; } }
                });

                if (extremeDanger) {
                    videoBox.style.borderColor = "#FF3333"; statusConsole.style.borderColor = "#FF3333";
                    statusConsole.innerText = `CRITICAL WARNING: Close ${targetObstacle.toUpperCase()} Ahead!`;
                    triggerBeep(950, 0.12);
                    let now = Date.now(); if (now - lastSpokenTime > 1800) { speak(`Warning. Close ${targetObstacle} detected.`); lastSpokenTime = now; }
                } else {
                    videoBox.style.borderColor = "#FFFFFF"; statusConsole.style.borderColor = "#00FF00";
                    statusConsole.innerText = predictions.length > 0 ? `Visible: ${predictions.map(p=>p.class).join(', ')}` : "Scanning... Path clear.";
                }
            } catch(e) {}
            animationFrameId = requestAnimationFrame(runObjectDetectionLoop);
        }

        function toggleObjectScanner() {
            if (!objectModel) {
                speak("System core is still processing initialization.");
                return;
            }
            if (!isScanningObjects) { isScanningObjects = true; document.getElementById('btn-object').style.backgroundColor = "var(--danger-color)"; speak("Scanning engaged."); runObjectDetectionLoop(); }
            else { isScanningObjects = false; document.getElementById('btn-object').style.backgroundColor = "#222"; videoBox.style.borderColor = "#FFFFFF"; statusConsole.style.borderColor = "#FFFF00"; statusConsole.innerText = "Scanner off."; speak("Scanning disengaged."); if (animationFrameId) cancelAnimationFrame(animationFrameId); }
        }

        async function runTextReading() {
            if (isScanningObjects) toggleObjectScanner();
            statusConsole.innerText = "Analyzing text..."; speak("Reading text. Hold still.");
            const ctx = ocrCanvas.getContext('2d'); ctx.drawImage(video, 0, 0, ocrCanvas.width, ocrCanvas.height);
            try {
                const result = await Tesseract.recognize(ocrCanvas, 'eng'); const cleanText = result.data.text.trim();
                if (cleanText.length > 0) { statusConsole.innerText = `Read: "${cleanText}"`; speak(`The text reads: ${cleanText}`); }
                else { statusConsole.innerText = "No text parsed."; speak("No text detected."); }
            } catch (err) { speak("Text processing failure."); }
        }

        function runVoiceAssistant() {
            const Speech = window.SpeechRecognition || window.webkitSpeechRecognition;
            if (!Speech) { speak("Voice recognition unsupported on this browser. Try Chrome."); return; }
            const rec = new Speech(); rec.lang = 'en-US';
            statusConsole.innerText = "Listening..."; speak("How can I help you?");
            rec.onresult = function(e) {
                const phrase = e.results[0][0].transcript.toLowerCase();
                statusConsole.innerText = `Heard: "${phrase}"`;
                if (phrase.includes("object") || phrase.includes("scan") || phrase.includes("see")) toggleObjectScanner();
                else if (phrase.includes("read") || phrase.includes("text")) runTextReading();
                else if (phrase.includes("help") || phrase.includes("emergency")) triggerEmergencySOS();
                else { speak("Unknown command."); }
            };
        }

        function triggerEmergencySOS() {
            if (isScanningObjects) toggleObjectScanner();
            videoBox.style.borderColor = "#FF3333"; statusConsole.style.borderColor = "#FF3333";
            statusConsole.innerText = "EMERGENCY PROTOCOL ACTIVE."; triggerBeep(1100, 0.4);
            speak("Emergency sequence activated.");
        }

        document.getElementById('btn-voice').addEventListener('click', runVoiceAssistant);
        document.getElementById('btn-object').addEventListener('click', toggleObjectScanner);
        document.getElementById('btn-text').addEventListener('click', runTextReading);
        document.getElementById('btn-sos').addEventListener('click', triggerEmergencySOS);
        document.querySelectorAll('.btn').forEach(b => { b.addEventListener('focus', () => speak(b.getAttribute('aria-label'))); });
        
        window.addEventListener('keydown', (e) => {
            const c = e.key.toLowerCase();
            if (c === 'v') runVoiceAssistant(); if (c === 'o') toggleObjectScanner(); if (c === 't') runTextReading(); if (c === 's') triggerEmergencySOS();
        });

        initSystem(); startCamera();
    </script>
</body>
</html>
"""

components.html(ACCESSIBLE_UI_HTML, height=720, scrolling=False)
