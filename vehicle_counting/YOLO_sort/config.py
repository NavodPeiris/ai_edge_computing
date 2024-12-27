import cv2
import json

def draw_line(event, x, y, flags, param):
    global line_points, display_frame, scaling_factor
    if event == cv2.EVENT_LBUTTONDOWN:
        scaled_x, scaled_y = int(x / scaling_factor), int(y / scaling_factor)
        line_points.append((scaled_x, scaled_y))
        if len(line_points) == 2:
            cv2.line(display_frame, (int(line_points[0][0] * scaling_factor), int(line_points[0][1] * scaling_factor)),
                     (int(line_points[1][0] * scaling_factor), int(line_points[1][1] * scaling_factor)), (0, 255, 0), 2)

line_points = []
video_path = 'vehicles360.mp4'
config_file = 'config.json'

# Load the video and get a frame
cap = cv2.VideoCapture(video_path)
ret, frame = cap.read()
cap.release()

if not ret:
    print("Failed to load video.")
    exit()

# Automatically calculate scaling factor to fit frame in a fixed-size window
max_width, max_height = 800, 600  # Adjust as needed
frame_height, frame_width = frame.shape[:2]
scaling_factor = min(max_width / frame_width, max_height / frame_height)
display_frame = cv2.resize(frame, (int(frame_width * scaling_factor), int(frame_height * scaling_factor)))

cv2.namedWindow('Configure', cv2.WINDOW_NORMAL)
cv2.resizeWindow('Configure', max_width, max_height)
cv2.setMouseCallback('Configure', draw_line)

while True:
    temp_frame = display_frame.copy()
    if len(line_points) == 1:
        cv2.circle(temp_frame, (int(line_points[0][0] * scaling_factor), int(line_points[0][1] * scaling_factor)), 5, (0, 255, 0), -1)
    cv2.imshow('Configure', temp_frame)
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q') or len(line_points) == 2:
        break

cv2.destroyAllWindows()

config = {
    'counting_line': line_points,
    'vehicle_classes': [2, 3, 5]  # Classes for car, motorcycle, bus, truck = 7
}

with open(config_file, 'w') as file:
    json.dump(config, file)

print(f"Configuration saved to {config_file}")
