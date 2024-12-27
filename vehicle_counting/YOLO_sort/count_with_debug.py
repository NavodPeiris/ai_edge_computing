from ultralytics import YOLO
import cv2
import numpy as np
import json
import time
from sort import Sort

def load_config(config_file):
    with open(config_file, 'r') as file:
        return json.load(file)

def scale_line(original_line, original_res, model_res):
    """
    Scale line coordinates from original resolution to model's resized resolution.
    """
    scale_x = model_res[0] / original_res[0]
    scale_y = model_res[1] / original_res[1]
    return [
        (int(original_line[0][0] * scale_x), int(original_line[0][1] * scale_y)),
        (int(original_line[1][0] * scale_x), int(original_line[1][1] * scale_y))
    ]

def is_crossing_line(point, prev_point, line, min_movement=5):
    """
    Check if a moving point has crossed a line using vector cross product.
    Added minimum movement threshold to prevent false crossings.
    """
    if prev_point is None:
        return 0
    
    # Calculate movement distance
    movement = np.sqrt((point[0] - prev_point[0])**2 + (point[1] - prev_point[1])**2)
    if movement < min_movement:
        return 0
        
    line_vec = (line[1][0] - line[0][0], line[1][1] - line[0][1])
    point_vec = (point[0] - prev_point[0], point[1] - prev_point[1])
    
    # Cross product
    cross_product = line_vec[0] * point_vec[1] - line_vec[1] * point_vec[0]
    
    # Check if point is near the line segment
    line_length = np.sqrt(line_vec[0]**2 + line_vec[1]**2)
    if line_length == 0:
        return 0
        
    dist_to_line = abs(cross_product) / line_length
    if dist_to_line > 50:  # Adjust this threshold based video
        return 0
    
    if cross_product > 0:
        return 1
    elif cross_product < 0:
        return -1
    return 0

def count_vehicles(video_path, config_file, output_file, show_debug=True):
    config = load_config(config_file)
    model = YOLO('yolov8s.pt')
    
    # Adjust tracker parameters for better stability
    tracker = Sort(max_age=20,  # Increased from 10 to maintain ID longer
                  min_hits=5,   # Increased from 3 to require more hits before tracking
                  iou_threshold=0.4)  # Increased from 0.3 for better tracking

    cap = cv2.VideoCapture(video_path)
    original_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    original_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    original_res = (original_width, original_height)

    counts = {'in': 0, 'out': 0}
    counted_ids = set()
    previous_centers = {}
    
    # Add positions history for smoothing
    position_history = {}
    history_length = 5

    frame_count = 0
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1
        
        # Skip frames to reduce processing and smooth tracking
        if frame_count % 2 != 0:
            continue

        results = model.predict(frame, conf=0.35)  # Lowered confidence threshold
        detections = []

        model_res = (results[0].orig_shape[1], results[0].orig_shape[0])
        scaled_line = scale_line(config['counting_line'], original_res, model_res)

        for result in results:
            boxes = result.boxes
            for box in boxes:
                if int(box.cls) in config['vehicle_classes']:
                    x1, y1, x2, y2 = box.xyxy[0][:4].tolist()
                    conf = float(box.conf)
                    
                    # Filter out small detections that might be false positives
                    box_width = x2 - x1
                    box_height = y2 - y1
                    min_size = 30  # Adjust based on video
                    if box_width < min_size or box_height < min_size:
                        continue
                        
                    detections.append([x1, y1, x2, y2, conf])

        if len(detections) == 0:
            tracked_objects = tracker.update(np.empty((0, 5)))
        else:
            tracked_objects = tracker.update(np.array(detections))

        if show_debug:
            debug_frame = frame.copy()
            cv2.line(debug_frame, 
                    (scaled_line[0][0], scaled_line[0][1]),
                    (scaled_line[1][0], scaled_line[1][1]),
                    (0, 255, 0), 2)

        for obj in tracked_objects:
            x1, y1, x2, y2, track_id = obj.astype(int)
            current_center = (int((x1 + x2) / 2), int((y1 + y2) / 2))
            
            # Update position history
            if track_id not in position_history:
                position_history[track_id] = []
            position_history[track_id].append(current_center)
            if len(position_history[track_id]) > history_length:
                position_history[track_id].pop(0)
            
            # Use smoothed position
            if len(position_history[track_id]) >= 3:
                smoothed_center = tuple(np.mean(position_history[track_id], axis=0).astype(int))
            else:
                smoothed_center = current_center
            
            prev_center = previous_centers.get(track_id)
            
            if track_id not in counted_ids:
                crossing = is_crossing_line(smoothed_center, prev_center, scaled_line)
                if crossing != 0:
                    if crossing > 0:
                        counts['in'] += 1
                    else:
                        counts['out'] += 1
                    counted_ids.add(track_id)
            
            previous_centers[track_id] = smoothed_center
            
            if show_debug:
                cv2.rectangle(debug_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(debug_frame, f"ID: {track_id}", (x1, y1-10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        # Cleanup old tracks
        current_track_ids = set(obj[4] for obj in tracked_objects)
        previous_centers = {k: v for k, v in previous_centers.items() if k in current_track_ids}
        position_history = {k: v for k, v in position_history.items() if k in current_track_ids}

        if show_debug:
            cv2.putText(debug_frame, f"In: {counts['in']} Out: {counts['out']}", 
                       (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.imshow('Debug', debug_frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    cap.release()
    if show_debug:
        cv2.destroyAllWindows()

    with open(output_file, 'w') as file:
        json.dump(counts, file)

if __name__ == "__main__":
    # Record start time
    start_time = time.time()
    count_vehicles('vehicles360.mp4', 'config.json', 'output.json', show_debug=False)
    # Record end time
    end_time = time.time()

    # Calculate elapsed time
    elapsed_time = end_time - start_time
    print(f"Elapsed time: {elapsed_time} seconds")