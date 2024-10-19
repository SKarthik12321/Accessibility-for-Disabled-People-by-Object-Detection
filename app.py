import random
import streamlit as st
from PIL import Image
import time
import threading
import cv2
import numpy as np
from gtts import gTTS
from playsound import playsound
import os

# Set the page configuration
st.set_page_config(page_title="Object Detection for Accessibility", layout="centered")

# Function for speech using gTTS
def speak(text):
    tts = gTTS(text=text, lang='en')
    filename = "detected_objects.mp3"
    tts.save(filename)
    playsound(filename)
    os.remove(filename)

# Function to load YOLO model
def load_yolo_model():
    weights_file = "config_files/yolov3.weights"
    config_file = "config_files/yolov3.cfg"
    class_file = "config_files/coco.names"

    net = cv2.dnn.readNet(weights_file, config_file)
    
    layer_names = net.getLayerNames()
    output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]
    
    with open(class_file, 'r') as f:
        class_names = [line.strip() for line in f.readlines()]

    return net, output_layers, class_names

# Function to detect objects in the frame
def detect_objects(frame, net, output_layers, class_names):
    blob = cv2.dnn.blobFromImage(frame, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
    net.setInput(blob)
    layer_outputs = net.forward(output_layers)

    height, width, _ = frame.shape
    boxes, confidences, class_ids = [], [], []

    for output in layer_outputs:
        for detection in output:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > 0.5:
                center_x, center_y, w, h = (detection[0:4] * np.array([width, height, width, height])).astype(int)
                x, y = int(center_x - w / 2), int(center_y - h / 2)
                boxes.append([x, y, w, h])
                confidences.append(float(confidence))
                class_ids.append(class_id)

    indices = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)

    detected_objects = []
    if len(indices) > 0:
        for i in indices.flatten():
            label = str(class_names[class_ids[i]])
            detected_objects.append(label)
            cv2.rectangle(frame, (boxes[i][0], boxes[i][1]), (boxes[i][0] + boxes[i][2], boxes[i][1] + boxes[i][3]), (0, 255, 0), 2)
            cv2.putText(frame, f"{label} {confidences[i]:.2f}", (boxes[i][0], boxes[i][1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

    return detected_objects, frame

# Function to display motivational quotes
def get_random_quote():
    quotes = [
        "The only limit to our realization of tomorrow is our doubts of today.",
        "Believe you can and you're halfway there.",
        "You are stronger than you think.",
        "The best way to predict the future is to invent it.",
        "Believe in yourself and all that you are."
    ]
    return random.choice(quotes)

# Main function with UI enhancements and inlined CSS
def main():
    # Apply custom CSS in the app
    st.markdown("""
    <style>
        /* Center title and add background color */
        div[role="heading"] {
            text-align: center;
            background-color: #0d6efd;
            color: white;
            padding: 20px;
            border-radius: 10px;
            font-family: 'Courier New', Courier, monospace;
        }

        /* Gradient effect for the quote */
        .quote-style {
            text-align: center;
            font-size: 22px;
            color: #333;
            margin-top: 20px;
            font-family: 'Georgia', serif;
            background: linear-gradient(to right, #ff416c, #ff4b2b, #ff9068, #ff416c);
            color: transparent;
            -webkit-background-clip: text;
            background-clip: text;
        }

        /* Instruction block styling with gradient background */
        .instructions {
            background: linear-gradient(to right, #36d1dc, #5b86e5, #a1c4fd);
            padding: 20px;
            border-radius: 12px;
            text-align: center;
            margin: 20px auto;
            max-width: 750px;
            border: 2px solid #0d6efd;
            color: white;
            box-shadow: 0 4px 8px rgba(0, 0, 0, 0.1);
        }

        /* Style unordered list for instructions */
        ul {
            list-style-type: none;
            padding-left: 0;
            font-size: 18px;
            text-align: left;
        }

        ul li {
            padding: 5px 0;
        }

        /* Button styling */
        button {
            font-size: 18px;
            padding: 10px;
            background-color: #0d6efd;
            color: white;
            border: none;
            border-radius: 8px;
            transition: 0.3s;
        }

        button:hover {
            background-color: #0056b3;
        }

        /* Blockquote for quotes */
        blockquote {
            font-size: 20px;
            color: #333;
            text-align: center;
            margin-top: 10px;
        }
    </style>
    """, unsafe_allow_html=True)

    # Title and header centered with background color
    st.markdown("""
        <div role="heading">
            🌟 Object Detection App for Accessibility 🌟
        </div>
    """, unsafe_allow_html=True)

    # Display a centered quote with gradient color
    st.markdown(f"""
        <div class="quote-style">
            <em>💬 Quote of the Day: "{get_random_quote()}"</em>
        </div>
    """, unsafe_allow_html=True)
    
    # Instructions block with gradient background
    st.markdown("""
    <div class="instructions">
        <h3>How to Use:</h3>
        <ul>
            <li>🎥 <strong>Start the camera</strong> by clicking the button below.</li>
            <li>🔊 <strong>Detected objects</strong> will be spoken aloud every 3 seconds.</li>
            <li>🔈 Ensure your <strong>speakers are on</strong> for a full experience!</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)
    
    # Option to start or stop object detection
    start_button = st.button("🎥 Start Object Detection")
    stop_button = st.button("⏹️ Stop Object Detection")
    
    # Show an image placeholder and progress bar
    frame_placeholder = st.empty()
    detection_progress = st.progress(0)
    
    # Load YOLO model
    net, output_layers, class_names = load_yolo_model()
    cap = None

    last_speech_time = time.time()

    if start_button:
        cap = cv2.VideoCapture(0)
        st.success("Camera started. Detecting objects...")
    
    if stop_button and cap:
        cap.release()
        cv2.destroyAllWindows()
        st.warning("Camera stopped.")

    # Loop to keep detecting objects while the camera is active
    if cap and cap.isOpened():
        while True:
            ret, frame = cap.read()
            if not ret:
                st.error("Failed to grab frame")
                break

            # Detect objects
            detected_objects, frame = detect_objects(frame, net, output_layers, class_names)

            # Show the video feed
            frame_placeholder.image(frame, channels="BGR", use_column_width=True)
            
            # Update the progress bar as a visual feedback
            detection_progress.progress(min(100, int((time.time() - last_speech_time) * 33)))  # updates per 3 seconds

            # Speak detected objects every 3 seconds
            if time.time() - last_speech_time >= 3:
                if detected_objects:
                    object_str = ', '.join(detected_objects)
                    speech_output = f"Detected: {object_str}."
                    threading.Thread(target=speak, args=(speech_output,)).start()
                last_speech_time = time.time()

            # Allow the user to stop detection
            if stop_button:
                break

        cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()