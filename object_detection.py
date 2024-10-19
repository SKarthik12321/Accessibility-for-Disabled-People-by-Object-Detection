# object_detection.py
import cv2

# Load COCO class labels
classFile = "config_files/coco.names"
with open(classFile, 'rt') as f:
    classNames = f.read().rstrip('\n').split('\n')

# Load the model files
configPath = "config_files/ssd_mobilenet_v3_large_coco.pbtxt"
weightsPath = "config_files/frozen_inference_graph.pb"

# Initialize the DNN model
net = cv2.dnn_DetectionModel(weightsPath, configPath)
net.setInputSize(320, 320)
net.setInputScale(1.0/127.5)
net.setInputMean((127.5, 127.5, 127.5))
net.setInputSwapRB(True)

def detect_objects(frame):
    # Perform object detection
    classIds, confs, bbox = net.detect(frame, confThreshold=0.5)
    
    # Draw bounding boxes and labels on the frame
    for classId, confidence, box in zip(classIds.flatten(), confs.flatten(), bbox):
        cv2.rectangle(frame, box, color=(0, 255, 0), thickness=2)
        cv2.putText(frame, classNames[classId - 1], (box[0] + 10, box[1] + 30),
                    cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 2)
    
    detected_labels = [classNames[classId - 1] for classId in classIds.flatten()]
    return frame, detected_labels