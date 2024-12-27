import cv2
import numpy as np
import json
from sort import Sort
from ultralytics import YOLO
import time
import os
from datetime import datetime

# Initialize YOLO model once
yolo_model = YOLO("Weights/yolov8n.pt")

# Define class labels
class_labels = [
    "person", "bicycle", "car", "motorbike", "aeroplane", "bus", "train", "truck", "boat",
    "traffic light", "fire hydrant", "stop sign", "parking meter", "bench",
    "cat", "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe",
    "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite", 
    "baseball bat", "baseball glove", "skateboard", "surfboard", "tennis racket",
    "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl",
    "banana", "apple", "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake",
    "chair", "sofa", "pottedplant", "bed", "diningtable", "toilet", "tvmonitor", "laptop", "mouse", 
    "remote", "keyboard", "cell phone", "microwave", "oven", "toaster", "sink", "refrigerator", "book", 
    "clock", "vase", "scissors", "teddy bear", "hair drier", "toothbrush"
]

# Initialize tracker
tracker = Sort(max_age=20, min_hits=3, iou_threshold=0.3)

# Load region mask and check its validity
region_mask = cv2.imread("Media/mask360.png")
if region_mask is None:
    print("Mask image not found!")
    exit()

# Read the JSON file
with open('config.json', 'r') as file:
    config = json.load(file)

# Extract the counting line coordinates
counting_line = config["counting_line"]
x1, y1 = counting_line[0]
x2, y2 = counting_line[1]

# Define line limits for counting
count_line = [x1, y1, x2, y2]

# List to store counted IDs
counted_ids = []

# Initialize the count dictionary
vehicle_count = []

# Define output file path
output_json_path = "vehicle_count.json"

# Check if the output file exists; if it does, load the existing data
if os.path.exists(output_json_path):
    with open(output_json_path, "r") as json_file:
        vehicle_count = json.load(json_file)

# Video capture
video_path = "Media/vehicles360.mp4"
cap = cv2.VideoCapture(video_path)

# Resize mask to match frame size
ret, frame = cap.read()
if not ret:
    print("Error reading the video file.")
    cap.release()
    exit()

region_mask = cv2.resize(region_mask, (frame.shape[1], frame.shape[0]))

# Record start time
start = time.time()

# Process the video
while cap.isOpened():
    start_time = time.time()  # Track the time for each frame processing

    success, frame = cap.read()
    if not success:
        break

    # Apply region mask
    masked_frame = cv2.bitwise_and(frame, region_mask)

    # Perform object detection
    detection_results = yolo_model(masked_frame, stream=True)

    # Collect detections
    detection_array = np.empty((0, 5))

    for result in detection_results:
        for box in result.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            width, height = x2 - x1, y2 - y1
            confidence = round(box.conf[0].item(), 2)
            class_id = int(box.cls[0])
            class_name = class_labels[class_id]

            if class_name in ["car", "truck", "motorbike", "bus"] and confidence > 0.3:
                detection_entry = np.array([x1, y1, x2, y2, confidence])
                detection_array = np.vstack((detection_array, detection_entry))

    # Update tracker
    tracked_objects = tracker.update(detection_array)

    for obj in tracked_objects:
        x1, y1, x2, y2, obj_id = map(int, obj)
        width, height = x2 - x1, y2 - y1

        # Calculate center of the box
        center_x, center_y = x1 + width // 2, y1 + height // 2

        # Check if the vehicle crosses the counting line
        if count_line[0] < center_x < count_line[2] and count_line[1] - 20 < center_y < count_line[1] + 20:
            if obj_id not in counted_ids:
                counted_ids.append(obj_id)
                timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                vehicle_count.append({"count": len(counted_ids), "timestamp": timestamp})

                # Save the count with timestamp to JSON after each vehicle cross
                with open(output_json_path, "w") as json_file:
                    json.dump(vehicle_count, json_file, indent=4)

    # Calculate frame processing time
    end_time = time.time()
    print(f"Time taken to process frame: {end_time - start_time:.2f} seconds")

# Record end time
end = time.time()

# Calculate elapsed time
elapsed_time = end - start
print(f"Elapsed time: {elapsed_time} seconds")

# Release video capture and cleanup
cap.release()

