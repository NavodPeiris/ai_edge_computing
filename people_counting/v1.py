import cv2
import numpy as np
import json
from pathlib import Path
from datetime import datetime

class SimpleCounter:
    def __init__(self, video_path, interval_seconds=5, debug=True):
        self.video_path = video_path
        self.interval_seconds = interval_seconds
        self.debug = debug
        
        # Initialize YOLO
        self.net = cv2.dnn.readNet(
            "yolov4-tiny.weights",
            "yolov4-tiny.cfg"
        )
        self.net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
        self.net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
        
        # Load classes
        with open("coco.names", "r") as f:
            self.classes = f.read().strip().split("\n")
        
        self.output_layers = self.net.getUnconnectedOutLayersNames()
        self.results = []

    def detect_people(self, frame):
        height, width = frame.shape[:2]
        
        # Create blob from image
        blob = cv2.dnn.blobFromImage(frame, 1/255.0, (416, 416), 
                                   swapRB=True, crop=False)
        self.net.setInput(blob)
        
        # Get detections
        outs = self.net.forward(self.output_layers)
        
        # Lists for detected boxes
        boxes = []
        confidences = []
        
        # Process detections
        for out in outs:
            for detection in out:
                scores = detection[5:]
                class_id = np.argmax(scores)
                confidence = scores[class_id]
                
                # Filter for people with good confidence
                if confidence > 0.5 and self.classes[class_id] == "person":
                    center_x = int(detection[0] * width)
                    center_y = int(detection[1] * height)
                    w = int(detection[2] * width)
                    h = int(detection[3] * height)
                    x = int(center_x - w/2)
                    y = int(center_y - h/2)
                    
                    boxes.append([x, y, w, h])
                    confidences.append(float(confidence))
        
        # Apply non-maximum suppression
        if boxes:
            indices = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)
            return len(indices), boxes, indices
        return 0, [], []

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
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            
            frame_count += 1
            
            # Process every 3rd frame for speed
            if frame_count % 3 == 0:
                count, boxes, indices = self.detect_people(frame)
                max_count_in_interval = max(max_count_in_interval, count)
                
                # Debug visualization
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
            
            # Save interval results
            if frame_count % frames_per_interval == 0:
                interval_count += 1
                timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                self.results.append({
                    "interval": interval_count,
                    "timestamp": timestamp,
                    "frame_number": frame_count,
                    "max_people_count": max_count_in_interval
                })
                print(f"Interval {interval_count}: {max_count_in_interval} people")
                max_count_in_interval = 0
        
        # Save final results
        output_path = Path(self.video_path).stem + "_counts.json"
        with open(output_path, 'w') as f:
            json.dump({
                "video_path": self.video_path,
                "interval_seconds": self.interval_seconds,
                "total_intervals": len(self.results),
                "counts": self.results
            }, f, indent=4)
        
        cap.release()
        if self.debug:
            cv2.destroyAllWindows()
        
        print(f"\nResults saved to {output_path}")
        return self.results

if __name__ == "__main__":
    video_path = "video.mp4"
    counter = SimpleCounter(video_path, interval_seconds=5, debug=True)
    results = counter.count_people()