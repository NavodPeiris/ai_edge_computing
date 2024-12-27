import json
import numpy as np
import supervision as sv
from ultralytics import YOLO

# Load YOLOv8 model
print("Loading YOLOv8 model...")
model = YOLO("yolov8n.pt")

# Map class names
# contains mapping of class IDs
CLASS_NAMES_DICT = model.model.names
# 2 - car, 3 - motorcycle, 5 - bus, 7 - truck
SELECTED_CLASS_NAMES = ['car', 'motorcycle', 'bus', 'truck']
SELECTED_CLASS_IDS = [
    {value: key for key, value in CLASS_NAMES_DICT.items()}[class_name]
    for class_name in SELECTED_CLASS_NAMES
]

# Input and output video path
SOURCE_VIDEO_PATH = "vehicles360.mp4"
TARGET_VIDEO_PATH = "result.mp4"

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

# Initialize ByteTracker
print("Initializing trackers and annotators...")
byte_tracker = sv.ByteTrack(
    track_activation_threshold=0.25,
    lost_track_buffer=30,
    minimum_matching_threshold=0.8,
    frame_rate=30,
    minimum_consecutive_frames=3
)
byte_tracker.reset()

# Initialize video info and frame generator
# VideoInfo(width=3840, height=2160, fps=25, total_frames=538)
video_info = sv.VideoInfo.from_video_path(SOURCE_VIDEO_PATH)
generator = sv.get_video_frames_generator(SOURCE_VIDEO_PATH)

# Calculate scaling factor based on video resolution
original_width = 3840
original_height = 2160
video_width, video_height = video_info.width, video_info.height

# Compute scale factors
scale_x = video_width / original_width
scale_y = video_height / original_height
scale_factor = min(scale_x, scale_y)  # Use the smaller scale factor to maintain aspect ratio

# Define a minimum thickness to avoid invalid thickness values
MIN_THICKNESS = 1

# Initialize annotators with dynamic scaling based on resolution
box_thickness = max(int(4 * scale_factor), MIN_THICKNESS)
text_thickness = max(int(2 * scale_factor), MIN_THICKNESS)
text_scale = 1.5 * scale_factor
line_zone_thickness = max(int(4 * scale_factor), MIN_THICKNESS)
line_zone_text_thickness = max(int(4 * scale_factor), MIN_THICKNESS)
line_zone_text_scale = 2 * scale_factor

# Initialize annotators
line_zone = sv.LineZone(start=LINE_START, end=LINE_END)
box_annotator = sv.BoxAnnotator(thickness=box_thickness)
label_annotator = sv.LabelAnnotator(text_thickness=text_thickness, text_scale=text_scale, text_color=sv.Color.BLACK)
trace_annotator = sv.TraceAnnotator(thickness=max(4 * scale_factor, MIN_THICKNESS), trace_length=50)
line_zone_annotator = sv.LineZoneAnnotator(
    thickness=line_zone_thickness,
    text_thickness=line_zone_text_thickness,
    text_scale=line_zone_text_scale
)

# Callback function for video processing
def callback(frame: np.ndarray, index: int) -> np.ndarray:
    results = model(frame, verbose=False)[0]
    detections = sv.Detections.from_ultralytics(results)
    detections = detections[np.isin(detections.class_id, SELECTED_CLASS_IDS)]
    detections = byte_tracker.update_with_detections(detections)

    labels = [
        f"#{tracker_id} {model.model.names[class_id]} {confidence:0.2f}"
        for confidence, class_id, tracker_id in zip(detections.confidence, detections.class_id, detections.tracker_id)
    ]

    annotated_frame = frame.copy()
    annotated_frame = trace_annotator.annotate(scene=annotated_frame, detections=detections)
    annotated_frame = box_annotator.annotate(scene=annotated_frame, detections=detections)
    annotated_frame = label_annotator.annotate(scene=annotated_frame, detections=detections, labels=labels)
    line_zone.trigger(detections)

    return line_zone_annotator.annotate(annotated_frame, line_counter=line_zone)

# Process and save video
print("Processing video and saving results...")
sv.process_video(
    source_path=SOURCE_VIDEO_PATH,
    target_path=TARGET_VIDEO_PATH,
    callback=callback
)
print(f"Video saved at: {TARGET_VIDEO_PATH}")
