# Status: Work in Progress

- OpenCV method - Fastest but needs to improve accuracy
- YOLOv8 with SORT
- YOLOv8 with Supervision - Most accurate, but slow when using CPU

* Test video can be found using link in the **Test Video link.txt** file.

## OpenCV Vehicle Counting Method

### Overview

This method uses **background subtraction** and **contour detection** to identify and count vehicles in a video. It tracks vehicles by matching them across frames and counts the number of vehicles that cross predefined counting lines.

### Key Steps

1. **Vehicle Detection**:

   - The video is processed frame by frame using OpenCV's **BackgroundSubtractorMOG2**.
   - Moving objects are detected by subtracting the background, and contours are analyzed to identify vehicles based on their size and aspect ratio.

2. **Tracking Vehicles**:

   - Detected vehicles are tracked across frames by associating new detections with existing tracks based on proximity.

3. **Counting Vehicles**:

   - The method uses predefined **counting lines** (specified in a config file) to track when vehicles cross them.
   - The direction of the vehicle's movement (in or out) is determined, and the counts are updated.

4. **Results**:
   - Counts for vehicles entering and exiting each line are saved in a JSON file.
   - The total processing time and number of frames processed are also recorded.

### Creating `traffic_config.json`

The `traffic_config.json` file is used to define counting lines and regions of interest (ROI) for vehicle counting.

### Steps to Create the Configuration File

1. **Run the Configuration Script**:

   - Run the script to open a video and start the configuration interface:

2. **Draw Counting Lines**:

   - Click and drag to draw lines in the video frame. Each line will be numbered automatically.
   - These lines are used to count vehicles entering or exiting specific regions.

3. **Define Region of Interest (ROI)**:

   - **NOT UTILIZED YET**
   - Press `r` to switch to ROI mode, then click to place points that will be connected to form a polygon.
   - The ROI defines areas of the video where vehicle detection is important.

4. **Other Controls**:
   - **`r`**: Toggle between drawing lines and defining ROI.
   - **`c`**: Clear all configurations (lines and ROI).
   - **`s`**: Save the configuration to `traffic_config.json`.
   - **`q`**: Quit the configuration interface.

---

## Vehicle Counting Model - YOLO with SORT

### Overview

The system detects and tracks vehicles in real-time using YOLOv8 for object detection and SORT for tracking. The vehicle count is based on the number of vehicles that cross a specified line, and the system handles vehicles entering and exiting.

### Key Steps

1. **Vehicle Detection (YOLO)**: YOLOv8 detects vehicles in each video frame. Only vehicles (based on specified classes) are considered for tracking.
2. **Tracking (SORT)**: The SORT algorithm assigns unique IDs to detected vehicles and tracks their movement across frames.
3. **Counting**: A predefined counting line is drawn, and when a vehicle crosses this line (in either direction), it is counted. Movement smoothing is applied to reduce false crossings.
4. **Output**: The total number of vehicles that have crossed the line in each direction ("in" and "out") is saved in a JSON file.

### Creating `config.json`

### Steps to Create the Configuration File

1. **Run the Configuration Script**:

   - Run the script to open a video and start the configuration interface:

2. **Draw Counting Lines**:

   - Click and drag to draw lines in the video frame. Each line will be numbered automatically.
   - These lines are used to count vehicles entering or exiting.

3. **Includes defining vehicle classes**

---

### YOLO with SORT and Regions

#### Overview:

Uses **YOLOv8** for object detection combined with the **SORT** algorithm to track vehicles. A **region mask** is applied to focus on specific areas of the video, filtering out irrelevant sections. Vehicles are counted when they cross a predefined counting line.

#### Key Steps:

1. **YOLOv8 Detection**:

   - The YOLOv8 model is used to detect vehicles in each frame.
   - Detected objects are filtered based on a confidence threshold and the vehicle class

2. **Region of Interest (ROI)**:

   - A custom region mask is applied to focus the detection on specific areas of the frame, ignoring irrelevant parts
   - The mask is resized to match the video frame dimensions.

3. **Object Tracking with SORT**:

   - The SORT algorithm assigns unique IDs to tracked vehicles and follows them across frames.
   - A bounding box is assigned to each detected vehicle, and the tracker updates the positions as the video progresses.

4. **Counting Mechanism**:

   - A counting line is defined, and vehicles are counted when they cross it. Each crossing vehicle is checked by its center position against the line.
   - Once a vehicle crosses the line, its ID is recorded to avoid double-counting.

5. **Data Output**:
   - Vehicle count is saved in a JSON file along with a timestamp after each crossing event.

Here’s a more concise version of the README section for the **"Config Generation Script"**:

### Config Generation Script

Manually define the counting line for vehicle counting in a video by selecting two points using the mouse

#### Key Steps:

1. **Draw Counting Line**: Click two points on the video frame to define the counting line.
2. **Save Configuration**: The coordinates are saved in `config.json` in scaled form.

---

## YOLO with Supervision

### Overview

This method uses **Ultralytics** for vehicle detection combined with the **Supervision** library for counting vehicles passing through a predefined counting line. Use tracking mechanism (ByteTrack) and a custom line zone for more precise counting of vehicles in/out of a specific area.

### Key Steps

1. **Model Loading**: The YOLOv8 model is loaded to detect vehicles.

2. **Counting Line Setup**: The coordinates of the counting line are extracted from a JSON configuration file. A line zone is created using these coordinates to monitor when vehicles cross the line.

3. **Vehicle Detection**: For each frame of the video, YOLO detects vehicles, and only the specified vehicle classes are considered for counting.

4. **Tracking with ByteTrack**: Detections are passed through a ByteTrack tracker that maintains vehicle tracking across frames. This ensures that vehicles are counted once, even if they temporarily disappear from the frame.

5. **Counting**: Vehicles are counted when they cross the counting line, with separate counts for vehicles going in and out. These counts are updated in real time.

6. **Output**: The vehicle counts are saved in a JSON file.

### Config Generation Script

Manually define the counting line for vehicle counting in a video by selecting two points using the mouse

#### Key Steps:

1. **Draw Counting Line**: Click two points on the video frame to define the counting line.
2. **Save Configuration**: The coordinates are saved in `config.json` in scaled form.
