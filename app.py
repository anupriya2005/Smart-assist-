import os
from flask import Flask, render_template_string, send_from_directory

app = Flask(__name__)

# Single-file architecture: Directly rendering the HTML template via string injection
HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>SMART ASSIST - Proximity Guard</title>
    
    <script src="https://cdn.jsdelivr.net/npm/@tensorflow/tfjs"></script>
    <script src="https://cdn.jsdelivr.net/npm/@tensorflow/@tensorflow-models/coco-ssd"></script>

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
            margin-bottom: 20px;
        }

        h1 { font-size: 2.8rem; color: var(--accent-color); margin: 0; }
        
        .main-container {
            display: flex;
            flex-direction: column;
            align-items: center;
            max-width: 700px;
            margin: 0 auto;
        }

        .video-box {
            position: relative;
            width: 100%;
            border: 5px solid var(--text-color);
            background-color: #111;
            margin-bottom: 20px;
            border-radius: 8px;
            overflow: hidden;
        }

        video {
            width: 100%;
            height: auto;
            display: block;
            transform: scaleX(-1); /* Mirror mode */
        }

        /* Large Accessible Trigger Button */
        .btn-toggle {
            background-color: #222;
            color: var(--accent-color);
            border: 4px solid var(--accent-color);
            padding: 25px 40px;
            font-size: 1.8rem;
            font-weight: bold;
            cursor: pointer;
            border-radius: 12px;
            width: 100%;
            transition: all 0.2s ease;
        }

        .btn-toggle:focus, .btn-toggle:hover {
            background-color: var(--accent-color);
            color: #000;
            outline: 4px solid var(--safe-color);
        }

        .btn-active {
            border-color: var(--danger-color);
            color: var(--danger-color);
            animation: pulse 1.5s infinite;
        }

        /* Dynamic Status Board */
        #status-console {
            margin-top: 20px;
            padding: 20px;
            border: 3px dashed var(--accent-color);
            font-size: 1.8rem;
            width: 90%;
            background-color: #111;
            font-weight: bold;
        }

        @keyframes pulse {
            0% { opacity: 1; }
            50% { opacity: 0.6; }
            100% { opacity: 1; }
        }
    </style>
</head>
<body>

    <header>
        <h1>SMART ASSIST</h1>
        <p style="color: var(--safe-color); font-size: 1.2rem; margin: 5px 0 0 0;">
            Real-Time Edge Detection & Proximity Warnings Active
        </p>
    </header>

    <div class="main-container">
        <div class="video-box" id="v-box">
            <video id="webcam" autoplay playsinline muted></video>
        </div>

        <button class="btn-toggle" id="start-btn" aria-label="Toggle Smart Scanner System. Status: Deactivated.">
            START ASSIST SCANNER
        </button>

        <div id="status-console" role="status" aria-live="assertive">
            System ready. Click button or press spacebar to start scanning.
        </div>
    </div>

    <script>
        const video = document.getElementById('webcam');
        const startBtn = document.getElementById('start-btn');
        const statusConsole = document.getElementById('status-console');
        const videoBox = document.getElementById('v-box');
        
        let model = null;
        let isScanning = false;
        let animationFrameId = null;
        let lastSpokenTime = 0;

        // Audio Context configuration to generate authentic hardware warning Beeps
        const audioCtx = new (window.AudioContext || window.webkitAudioContext)();

        // 1. Initialize System and Load local COCO-SSD ML Architecture
        async function initSystem() {
            statusConsole.innerText = "Loading AI Core Models... Please wait.";
            try {
                model = await cocoSsd.load();
                statusConsole.innerText = "AI Core Ready. Press Start Scanner.";
                speak("System models initialized successfully.");
            } catch (err) {
                statusConsole.innerText = "Error loading intelligence models.";
                console.error(err);
            }
        }

        // 2. Hardware beep synthesizer 
        function triggerWarningBeep(frequency, duration) {
            if (audioCtx.state === 'suspended') {
                audioCtx.resume();
            }
            const oscillator = audioCtx.createOscillator();
            const gainNode = audioCtx.createGain();
            
            oscillator.type = 'sine';
            oscillator.frequency.value = frequency; // High pitch for immediate threat
            
            gainNode.gain.setValueAtTime(0.3, audioCtx.currentTime);
            gainNode.gain.exponentialRampToValueAtTime(0.01, audioCtx.currentTime + duration);
            
            oscillator.connect(gainNode);
            gainNode.connect(audioCtx.destination);
            
            oscillator.start();
            oscillator.stop(audioCtx.currentTime + duration);
        }

        // 3. Accessibility Native Screen-Reader TTS
        function speak(text) {
            window.speechSynthesis.cancel();
            const utterance = new SpeechSynthesisUtterance(text);
            utterance.rate = 1.1;
            window.speechSynthesis.speak(utterance);
        }

        // 4. Connect input Webcam Device
        async function setupWebcam() {
            if (navigator.mediaDevices && navigator.mediaDevices.getUserMedia) {
                try {
                    const stream = await navigator.mediaDevices.getUserMedia({
                        video: { width: 640, height: 480, facingMode: "environment" }
                    });
                    video.srcObject = stream;
                } catch (err) {
                    statusConsole.innerText = "Webcam input access blocked.";
                    speak("Webcam access error.");
                }
            }
        }

        // 5. Core AI Processing and Proximity Loop
        async function processingLoop() {
            if (!isScanning) return;

            // Detect objects inside the current frame
            const predictions = await model.detect(video);
            
            let threatDetected = false;
            let primaryObject = "";
            let maxProximityScore = 0;

            predictions.forEach(prediction => {
                const [x, y, width, height] = prediction.bbox;
                
                // Proximity calculations: Determine threat scale based on frame space consumption
                const frameArea = video.videoWidth * video.videoHeight;
                const objectArea = width * height;
                const proximityPercentage = (objectArea / frameArea) * 100;

                // Threshold: If an object occupies more than 28% of the viewport, it is too close!
                if (proximityPercentage > 28) {
                    threatDetected = true;
                    if (proximityPercentage > maxProximityScore) {
                        maxProximityScore = proximityPercentage;
                        primaryObject = prediction.class;
                    }
                }
            });

            if (threatDetected) {
                // Change UI container borders to bright warning Red
                videoBox.style.borderColor = "#FF3333";
                statusConsole.style.borderColor = "#FF3333";
                statusConsole.innerText = `WARNING: Close obstacle detected: ${primaryObject.toUpperCase()}!`;

                // Fire an immediate, physical hardware warning beep
                triggerWarningBeep(880, 0.15);

                // Throttle spoken words to once every 2 seconds to prevent audio overlapping
                let now = Date.now();
                if (now - lastSpokenTime > 2000) {
                    speak(`Warning. ${primaryObject} is directly ahead.`);
                    lastSpokenTime = now;
                }
            } else {
                // Return interface to standard safe state colors
                videoBox.style.borderColor = "#FFFFFF";
                statusConsole.style.borderColor = "#00FF00";
                
                if (predictions.length > 0) {
                    statusConsole.innerText = `Path Clear. Detected: ${predictions[0].class}`;
                } else {
                    statusConsole.innerText = "Scanning... Path completely clear ahead.";
                }
            }

            // Keep looping over subsequent camera updates
            animationFrameId = requestAnimationFrame(processingLoop);
        }

        // 6. Controller triggers
        startBtn.addEventListener('click', () => {
            if (!isScanning) {
                isScanning = true;
                startBtn.innerText = "STOP ASSIST SCANNER";
                startBtn.classList.add('btn-active');
                startBtn.setAttribute('aria-label', "Toggle Smart Scanner System. Status: Activated.");
                speak("Scanner loop activated.");
                processingLoop();
            } else {
                isScanning = false;
                startBtn.innerText = "START ASSIST SCANNER";
                startBtn.classList.remove('btn-active');
                startBtn.setAttribute('aria-label', "Toggle Smart Scanner System. Status: Deactivated.");
                videoBox.style.borderColor = "#FFFFFF";
                statusConsole.style.borderColor = "#FFFF00";
                statusConsole.innerText = "Scanner halted.";
                speak("Scanner loop stopped.");
                if (animationFrameId) cancelAnimationFrame(animationFrameId);
            }
        });

        // Trigger loading patterns on boot up
        initSystem();
        setupWebcam();
    </script>
</body>
</html>
"""

@app.route('/')
def home():
    # Stream the app template dynamically directly to the client browser
    return render_template_string(HTML_TEMPLATE)

if __name__ == '__main__':
    # Initialize local Flask server infrastructure
    app.run(host='0.0.0.0', port=5000, debug=True)
