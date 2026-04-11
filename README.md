Smart Assist: AI-Powered Vision for the Visually Impaired
Smart Assist is a real-time assistive technology system that utilizes Deep Learning and Advanced Image Processing to provide spatial awareness for visually impaired individuals. It translates visual surroundings into auditory feedback, prioritizing safety through low-light enhancement and system-health monitoring.

Key Features:
Dual-Model Parallel Inference: Utilizes multi-threading to run concurrent YOLOv8 models, ensuring zero-latency object detection.

Adaptive Night Vision (CLAHE): Automatically detects low-light environments and applies Contrast Limited Adaptive Histogram Equalization to enhance visibility for the AI.

Spatial Intelligence: Provides real-time distance estimation and directional mapping (Left, Right, Front) based on geometric bounding box analysis.

System Health Monitoring: Actively monitors camera focus using Laplacian Variance; alerts the user via voice if the lens is blurred or occluded.

Non-Blocking TTS: Employs a threaded Text-to-Speech (TTS) engine, ensuring the video feed remains fluid while the system provides audio feedback.

Architecture:
The system follows a modular pipeline designed for edge deployment:

Ingestion: Raw frame capture via OpenCV.

Preprocessing: Laplacian focus check and CLAHE brightness correction.

Inference: Parallel YOLOv8 Nano models for object and hazard recognition.

Logic Engine: Spatial mapping and priority-based alert filtering.

Feedback: Non-blocking voice commands and hazard beeps.

Tech Stack:
Language: Python 3.x

AI Model: YOLOv8 (Ultralytics)

Vision Library: OpenCV, Supervision

Audio: Pyttsx3 (SAPI5/Espeak)

Concurrency: Python Threading & Daemon Processes

Social Impact (UN SDGs):
This project is aligned with the United Nations Sustainable Development Goals:

Goal 10: Reduced Inequalities: Providing affordable, high-tech assistive tools to empower people with disabilities.

Goal 11: Sustainable Cities and Communities: Enhancing accessibility in urban environments through AI-driven navigation.
