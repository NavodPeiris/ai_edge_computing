import cv2
import numpy as np
import json
import time
from collections import defaultdict

class VehicleCounter:
    def __init__(self, video_path, config_path):
        self.video_path = video_path
        self.load_config(config_path)
        
        # Get video properties
        cap = cv2.VideoCapture(video_path)
        self.frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        cap.release()
        
        # Adjust parameters based on video resolution
        area_factor = (self.frame_width * self.frame_height) / (1920 * 1080)  # Reference resolution
        self.min_contour_area = int(800 * area_factor)  # Minimum vehicle size
        self.max_contour_area = int(15000 * area_factor)  # Maximum vehicle size
        self.distance_threshold = int(50 * np.sqrt(area_factor))  # Tracking distance threshold
        
        # Vehicle tracking parameters
        self.tracked_vehicles = []
        
        # Counting data
        self.vehicles_in = defaultdict(int)
        self.vehicles_out = defaultdict(int)
        
        # Background subtractor with adjusted parameters
        self.bg_subtractor = cv2.createBackgroundSubtractorMOG2(
            history=500,
            varThreshold=24,  # Increased for better detection
            detectShadows=False
        )

    def load_config(self, config_path):
        """Load configuration from JSON file"""
        with open(config_path, 'r') as f:
            config = json.load(f)
        self.counting_lines = config['counting_lines']
        
        # Calculate line directions
        self.line_directions = []
        for line in self.counting_lines:
            x1, y1 = line[0]
            x2, y2 = line[1]
            dx = x2 - x1
            dy = y2 - y1
            self.line_directions.append((-dy, dx))

    def detect_vehicles(self, frame):
        """Detect vehicles using background subtraction and contour detection"""
        # Apply background subtraction
        fg_mask = self.bg_subtractor.apply(frame)
        
        # Enhanced noise reduction for 4K
        fg_mask = cv2.medianBlur(fg_mask, 7)  # Increased kernel size
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7))
        fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_CLOSE, kernel, iterations=3)
        
        # Find contours
        contours, _ = cv2.findContours(fg_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Process each contour
        vehicles = []
        for contour in contours:
            area = cv2.contourArea(contour)
            if self.min_contour_area < area < self.max_contour_area:
                x, y, w, h = cv2.boundingRect(contour)
                # Additional filtering based on aspect ratio
                aspect_ratio = w / h
                if 0.5 < aspect_ratio < 2.5:  # Typical vehicle aspect ratios
                    center_x = x + w // 2
                    center_y = y + h // 2
                    vehicles.append({
                        'center': (center_x, center_y),
                        'area': area,
                        'bbox': (x, y, w, h)
                    })
        
        return vehicles

    def update_tracking(self, vehicles):
        """Update vehicle tracking"""
        if not self.tracked_vehicles:
            for vehicle in vehicles:
                self.tracked_vehicles.append({
                    'centers': [vehicle['center']],
                    'counted': set(),
                    'last_seen': 0
                })
            return

        new_tracked_vehicles = []
        used_vehicles = set()

        # Update existing tracks
        for track in self.tracked_vehicles:
            if len(track['centers']) == 0:
                continue
                
            last_center = track['centers'][-1]
            closest_dist = float('inf')
            closest_vehicle = None
            closest_idx = None

            # Find the closest detected vehicle
            for i, vehicle in enumerate(vehicles):
                if i in used_vehicles:
                    continue
                    
                center = vehicle['center']
                dist = np.sqrt((center[0] - last_center[0])**2 + 
                             (center[1] - last_center[1])**2)
                
                if dist < self.distance_threshold:
                    if dist < closest_dist:
                        closest_dist = dist
                        closest_vehicle = vehicle
                        closest_idx = i

            if closest_vehicle is not None:
                used_vehicles.add(closest_idx)
                track['centers'].append(closest_vehicle['center'])
                track['last_seen'] = 0
                if len(track['centers']) > 10:
                    track['centers'] = track['centers'][-10:]
                new_tracked_vehicles.append(track)
            else:
                track['last_seen'] += 1
                if track['last_seen'] < 5:  # Keep track alive for 5 frames
                    new_tracked_vehicles.append(track)

        # Add new tracks
        for i, vehicle in enumerate(vehicles):
            if i not in used_vehicles:
                new_tracked_vehicles.append({
                    'centers': [vehicle['center']],
                    'counted': set(),
                    'last_seen': 0
                })

        self.tracked_vehicles = new_tracked_vehicles

    def check_line_crosses(self):
        """Check if vehicles have crossed counting lines"""
        for track in self.tracked_vehicles:
            if len(track['centers']) < 2:
                continue

            last_center = track['centers'][-1]
            prev_center = track['centers'][-2]

            for line_idx, (line, direction) in enumerate(zip(self.counting_lines, self.line_directions)):
                if line_idx in track['counted']:
                    continue

                x1, y1 = line[0]
                x2, y2 = line[1]

                if self.check_line_crossing(prev_center, last_center, (x1, y1), (x2, y2)):
                    movement_vector = (
                        last_center[0] - prev_center[0],
                        last_center[1] - prev_center[1]
                    )
                    
                    dot_product = (movement_vector[0] * direction[0] + 
                                 movement_vector[1] * direction[1])
                    
                    if dot_product > 0:
                        self.vehicles_in[line_idx] += 1
                    else:
                        self.vehicles_out[line_idx] += 1
                    
                    track['counted'].add(line_idx)

    def check_line_crossing(self, p1, p2, p3, p4):
        """Check if line segments intersect"""
        def ccw(A, B, C):
            return (C[1] - A[1]) * (B[0] - A[0]) > (B[1] - A[1]) * (C[0] - A[0])

        return ccw(p1, p3, p4) != ccw(p2, p3, p4) and ccw(p1, p2, p3) != ccw(p1, p2, p4)

    def process_video(self):
        """Process video and count vehicles"""
        cap = cv2.VideoCapture(self.video_path)
        if not cap.isOpened():
            raise Exception("Error opening video file")

        frame_count = 0
        start_time = time.time()

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            frame_count += 1
            if frame_count % 2 != 0:  # Process every other frame
                continue

            vehicles = self.detect_vehicles(frame)
            self.update_tracking(vehicles)
            self.check_line_crosses()

        cap.release()

        processing_time = time.time() - start_time
        results = {
            'vehicles_in': dict(self.vehicles_in),
            'vehicles_out': dict(self.vehicles_out),
            'processing_time': processing_time,
            'frames_processed': frame_count // 2
        }

        with open('counting_results.json', 'w') as f:
            json.dump(results, f, indent=4)

        return results

if __name__ == "__main__":
    counter = VehicleCounter('vehicles360.mp4', 'traffic_config.json')
    results = counter.process_video()
    
    print("\nVehicle Counting Results:")
    for line_idx in results['vehicles_in'].keys():
        print(f"\nLine {line_idx}:")
        print(f"Vehicles IN: {results['vehicles_in'][line_idx]}")
        print(f"Vehicles OUT: {results['vehicles_out'][line_idx]}")
    
    print(f"\nTotal processing time: {results['processing_time']:.2f} seconds")
    print(f"Frames processed: {results['frames_processed']}")