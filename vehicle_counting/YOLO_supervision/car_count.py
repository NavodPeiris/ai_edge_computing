import time
import numpy as np
import json
from ultralytics import YOLO
import supervision as sv

# Load YOLOv8 model
print("Loading YOLOv8 model...")
model = YOLO("yolov8n.pt")

# Map class names and select desired vehicle classes
CLASS_NAMES_DICT = model.model.names
SELECTED_CLASS_NAMES = ['car', 'motorcycle', 'bus', 'truck']
SELECTED_CLASS_IDS = [
    {value: key for key, value in CLASS_NAMES_DICT.items()}[class_name]
    for class_name in SELECTED_CLASS_NAMES
]

# Input video path
SOURCE_VIDEO_PATH = "vehicles360.mp4"

# Read the JSON file
with open('config.json', 'r') as file:
    config = json.load(file)

# Extract the counting line coordinates
counting_line = config["counting_line"]
start_coords = counting_line[0]
end_coords = counting_line[1]

# Create the Points using the extracted coordinates
LINE_START = sv.Point(start_coords[0], start_coords[1])
LINE_END = sv.Point(end_coords[0], end_coords[1])
line_zone = sv.LineZone(start=LINE_START, end=LINE_END)

# Initialize tracker
byte_tracker = sv.ByteTrack(
    track_activation_threshold=0.25,
    lost_track_buffer=30,
    minimum_matching_threshold=0.8,
    frame_rate=30,
    minimum_consecutive_frames=3
)
byte_tracker.reset()

# Video info and frame generator
video_info = sv.VideoInfo.from_video_path(SOURCE_VIDEO_PATH)
generator = sv.get_video_frames_generator(SOURCE_VIDEO_PATH)

# Record start time
start_time = time.time()

# Process video frames and count vehicles
print("Processing video for vehicle counts...")
for frame in generator:
    results = model(frame, verbose=False)[0]
    detections = sv.Detections.from_ultralytics(results)
    detections = detections[np.isin(detections.class_id, SELECTED_CLASS_IDS)]
    detections = byte_tracker.update_with_detections(detections)
    line_zone.trigger(detections)

# Store counts in a JSON file
vehicle_counts = {"in_count": line_zone.in_count, "out_count": line_zone.out_count}
output_json_path = "vehicle_counts2.json"
with open(output_json_path, "w") as json_file:
    json.dump(vehicle_counts, json_file, indent=4)

print(f"Vehicle counts saved in: {output_json_path}")

# Record end time
end_time = time.time()

# Calculate elapsed time
elapsed_time = end_time - start_time
print(f"Elapsed time: {elapsed_time} seconds")