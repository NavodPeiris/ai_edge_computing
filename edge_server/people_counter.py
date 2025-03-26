import cv2
import numpy as np
import json
from pathlib import Path
from datetime import datetime
import os

class SimpleCounter:
    def __init__(self, video_path, interval_seconds=5, debug=True):
        self.video_path = video_path
        self.interval_seconds = interval_seconds
        self.debug = debug
        
        # Initialize YOLO
        self.net = cv2.dnn.readNet(
            "people_counting_data/yolov4-tiny.weights",
            "people_counting_data/yolov4-tiny.cfg"
        )
        self.net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
        self.net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
        
        # Load classes
        with open("people_counting_data/coco.names", "r") as f:
            self.classes = f.read().strip().split("\n")
        
        self.output_layers = self.net.getUnconnectedOutLayersNames()
        self.results = []
        
        # Create output directory for frames
        self.output_dir = "people_counting_data"
        os.makedirs(self.output_dir, exist_ok=True)

    def detect_people(self, frame):
        height, width = frame.shape[:2]
        
        # Improved blob creation with better size handling
        # Using a larger size improves detection of small/distant people
        blob = cv2.dnn.blobFromImage(frame, 1/255.0, (608, 608), 
                                   swapRB=True, crop=False)
        self.net.setInput(blob)
        
        outs = self.net.forward(self.output_layers)
        
        boxes = []
        confidences = []
        
        # Improved detection processing
        for out in outs:
            for detection in out:
                scores = detection[5:]
                class_id = np.argmax(scores)
                confidence = scores[class_id]
                
                # Lowered confidence threshold to catch more potential detections
                # Instead filter them better in NMS
                if confidence > 0.3 and self.classes[class_id] == "person":
                    center_x = int(detection[0] * width)
                    center_y = int(detection[1] * height)
                    w = int(detection[2] * width)
                    h = int(detection[3] * height)
                    
                    # Improved edge detection by extending boxes slightly
                    x = max(0, int(center_x - w/2) - 5)
                    y = max(0, int(center_y - h/2) - 5)
                    w = min(width - x, w + 10)
                    h = min(height - y, h + 10)
                    
                    boxes.append([x, y, w, h])
                    confidences.append(float(confidence))
        
        # Improved NMS parameters for better handling of overlapping detections
        if boxes:
            indices = cv2.dnn.NMSBoxes(boxes, confidences, 0.3, 0.2)
            return len(indices), boxes, indices, confidences
        return 0, [], [], []

    def save_marked_frame(self, frame, boxes, indices, confidences, count, interval, timestamp):
        marked_frame = frame.copy()
        
        # Draw each detection with improved visibility
        for i in range(len(boxes)):
            if i in indices:
                x, y, w, h = boxes[i]
                # Draw box
                cv2.rectangle(marked_frame, (x, y), (x + w, y + h), (0, 255, 0), 3)
                # Draw confidence
                conf_text = f'{confidences[i]:.2f}'
                cv2.putText(marked_frame, conf_text, (x, y - 10), 
                          cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                # Draw ID number
                cv2.rectangle(marked_frame, (x+2, y+5), (x+30, y+30), (255, 255, 255), -1)
                cv2.putText(marked_frame, str(i+1), (x+5, y+25), 
                          cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
        
        # Draw total count
        cv2.rectangle(marked_frame, (5, 5), (250, 80), (0, 0, 0), -1)
        cv2.putText(marked_frame, f'Total Count: {count}', (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.putText(marked_frame, f'Interval: {interval}', (10, 70), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        frame_filename = f"interval_{interval}_count_{count}_{timestamp.replace(':', '-')}.jpg"
        frame_path = os.path.join(self.output_dir, frame_filename)
        cv2.imwrite(frame_path, marked_frame)
        return frame_path, marked_frame

    def count_people(self):
        cap = cv2.VideoCapture(self.video_path)
        if not cap.isOpened():
            raise Exception("Error opening video file")
        
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        frames_per_interval = int(fps * self.interval_seconds)
        
        print(f"Video FPS: {fps}")
        print(f"Frames per interval: {frames_per_interval}")
        print(f"Total frames: {total_frames}")
        
        frame_count = 0
        interval_count = 0
        max_count_in_interval = 0
        best_frame_data = None
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            
            frame_count += 1
            
            # Process every 4th frame
            if frame_count % 4 == 0:
                count, boxes, indices, confidences = self.detect_people(frame)
                
                # Frame with max count in that interval is chosen
                if count > max_count_in_interval:
                    max_count_in_interval = count
                    best_frame_data = (frame.copy(), boxes, indices, confidences)
                
                if self.debug:
                    debug_frame = frame.copy()
                    for i in range(len(boxes)):
                        if i in indices:
                            x, y, w, h = boxes[i]
                            cv2.rectangle(debug_frame, (x, y), 
                                        (x + w, y + h), (0, 255, 0), 2)
                    
                    cv2.putText(debug_frame, f'Count: {count}', (10, 30), 
                              cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                    cv2.imshow('Detection', debug_frame)
                    
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        break
            
            if frame_count % frames_per_interval == 0:
                interval_count += 1
                timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
                
                """
                if best_frame_data is not None:
                    frame, boxes, indices, confidences = best_frame_data
                    frame_path, _ = self.save_marked_frame(
                        frame, boxes, indices, confidences,
                        max_count_in_interval, interval_count, timestamp
                    )
                else:
                    frame_path = None
                """
                
                self.results.append({
                    "interval": interval_count,
                    "timestamp": timestamp,
                    "frame_number": frame_count,
                    "people_count": max_count_in_interval,
                    #"verification_frame": frame_path
                })
                
                print(f"Interval {interval_count}: {max_count_in_interval} people (Frame saved)")
                max_count_in_interval = 0
                best_frame_data = None
        
        output_path = "people_counts.json"
        with open(output_path, 'w') as f:
            json.dump({
                "video_path": self.video_path,
                "interval_seconds": self.interval_seconds,
                "total_intervals": len(self.results),
                "frames_directory": self.output_dir,
                "counts": self.results
            }, f, indent=4)
        
        cap.release()
        if self.debug:
            cv2.destroyAllWindows()
        
        print(f"\nResults saved to {output_path}")
        print(f"Verification frames saved in {self.output_dir}")
        return self.results


def count_people(video_path, interval_seconds):
    # Change debug=True to see the debug video
    counter = SimpleCounter(video_path, interval_seconds=interval_seconds, debug=False)
    results = counter.count_people()
    return results