import cv2
import json
import numpy as np

class ConfigCreator:
    def __init__(self, video_path):
        self.video_path = video_path
        self.frame = None
        self.display_frame = None
        self.lines = []
        self.roi_points = []
        self.drawing = False
        self.current_action = 'line'  # 'line' or 'roi'
        self.scale_factor = 1.0
        self.window_width = 1280  # maximum window width
        
    def get_frame(self):
        """Extract a frame from video for configuration"""
        cap = cv2.VideoCapture(self.video_path)
        ret, frame = cap.read()
        cap.release()
        
        if ret:
            # Calculate scaling factor to fit the frame within window_width
            h, w = frame.shape[:2]
            if w > self.window_width:
                self.scale_factor = self.window_width / w
                new_width = self.window_width
                new_height = int(h * self.scale_factor)
                frame = cv2.resize(frame, (new_width, new_height))
            return frame
        return None

    def scale_point_to_original(self, x, y):
        """Convert display coordinates back to original frame coordinates"""
        return (int(x / self.scale_factor), int(y / self.scale_factor))

    def mouse_callback(self, event, x, y, flags, param):
        if self.frame is None:
            return

        if event == cv2.EVENT_LBUTTONDOWN:
            self.drawing = True
            if self.current_action == 'line':
                self.lines.append([(x, y)])
            else:  # ROI mode
                self.roi_points.append((x, y))
                self.update_display()

        elif event == cv2.EVENT_MOUSEMOVE:
            if self.drawing and self.current_action == 'line':
                # Create a copy of the display frame for temporary line preview
                temp_frame = self.display_frame.copy()
                cv2.line(temp_frame, self.lines[-1][0], (x, y), (0, 255, 0), 2)
                
                # Add text showing line number
                line_num = len(self.lines)
                text_pos = (min(self.lines[-1][0][0], x), min(self.lines[-1][0][1], y) - 10)
                cv2.putText(temp_frame, f'Line {line_num}', text_pos, 
                          cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                
                cv2.imshow('Configuration', temp_frame)

        elif event == cv2.EVENT_LBUTTONUP:
            if self.drawing and self.current_action == 'line':
                self.drawing = False
                self.lines[-1].append((x, y))
                self.update_display()

    def update_display(self):
        """Update the display frame with all current configurations"""
        self.display_frame = self.frame.copy()
        
        # Draw all completed lines
        for i, line in enumerate(self.lines):
            if len(line) == 2:
                cv2.line(self.display_frame, line[0], line[1], (0, 255, 0), 2)
                # Add line number
                text_pos = (min(line[0][0], line[1][0]), min(line[0][1], line[1][1]) - 10)
                cv2.putText(self.display_frame, f'Line {i+1}', text_pos, 
                          cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        # Draw ROI points and connect them
        if len(self.roi_points) > 0:
            # Draw points
            for point in self.roi_points:
                cv2.circle(self.display_frame, point, 3, (0, 0, 255), -1)
            
            # Connect points to form polygon
            if len(self.roi_points) > 1:
                pts = np.array(self.roi_points, np.int32)
                cv2.polylines(self.display_frame, [pts], 
                            True if len(self.roi_points) > 2 else False, 
                            (0, 0, 255), 2)

        # Show current mode
        mode_text = f"Current Mode: {'ROI' if self.current_action == 'roi' else 'Line Drawing'}"
        cv2.putText(self.display_frame, mode_text, (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        cv2.imshow('Configuration', self.display_frame)

    def create_config(self):
        self.frame = self.get_frame()
        if self.frame is None:
            print("Error: Could not read video file")
            return
        
        self.display_frame = self.frame.copy()
        cv2.namedWindow('Configuration')
        cv2.setMouseCallback('Configuration', self.mouse_callback)
        
        print("\n=== Configuration Creator Instructions ===")
        print("1. Draw counting lines:")
        print("   - Click and drag to draw a line")
        print("   - Each line will be numbered automatically")
        print("\n2. Define Region of Interest (ROI):")
        print("   - Press 'r' to switch to ROI mode")
        print("   - Click to place points, they will be connected automatically")
        print("\n3. Other controls:")
        print("   - Press 'r' to toggle between Line and ROI modes")
        print("   - Press 'c' to clear all")
        print("   - Press 's' to save configuration")
        print("   - Press 'q' to quit")
        print("\nCurrent window scale factor:", self.scale_factor)
        
        self.update_display()
        
        while True:
            key = cv2.waitKey(1) & 0xFF
            
            if key == ord('r'):
                self.current_action = 'roi' if self.current_action == 'line' else 'line'
                print(f"Switched to {self.current_action} mode")
                self.update_display()
                
            elif key == ord('c'):
                self.frame = self.get_frame()
                self.display_frame = self.frame.copy()
                self.lines = []
                self.roi_points = []
                print("Cleared all configurations")
                self.update_display()
                
            elif key == ord('s'):
                # Convert coordinates back to original scale
                original_scale_lines = [
                    [self.scale_point_to_original(x, y) for x, y in line]
                    for line in self.lines
                ]
                original_scale_roi = [
                    self.scale_point_to_original(x, y)
                    for x, y in self.roi_points
                ]
                
                config = {
                    'counting_lines': original_scale_lines,
                    'roi_points': original_scale_roi
                }
                with open('traffic_config.json', 'w') as f:
                    json.dump(config, f)
                print("Configuration saved to traffic_config.json")
                
            elif key == ord('q'):
                break
                
        cv2.destroyAllWindows()

if __name__ == "__main__":
    config_creator = ConfigCreator('vehicles360.mp4')
    config_creator.create_config()