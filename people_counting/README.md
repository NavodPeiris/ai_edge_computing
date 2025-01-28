# People Counter with YOLO

Uses the YOLO to count the number of people in a given video file. The script processes the video at regular intervals and outputs a JSON file with the people count per interval, as well as saves frames with bounding boxes drawn around detected people.

## Version History

### v1:

- Initial version. Detects people in a video at specified intervals.
- Outputs a JSON file with people counts per interval.
- Does not save frames for verification.
- Processing is done to every 3rd frame.

### v2:

- **Frame Saving**: Adds functionality to save frames with bounding boxes, confidence scores, and person IDs.
- **Verification Frames**: The frame with the highest detected people count during each interval is saved as a verification frame.
- **Debugging Enhancements**: More details are drawn on the frames, including confidence scores and ID numbers for detected people.
- **Output Directory**: A dedicated directory (`_frames`) is created to store the saved frames, improving organization.
- **Improved Interval Processing**: The frame with the highest count in each interval is saved, ensuring better accuracy.

### v3:

- **Improved Blob Size**: The blob size is updated to (608, 608) to better handle small or distant people, improving detection quality.
- **Confidence Threshold Adjustment**: The confidence threshold is lowered to 0.3 to catch more potential detections, with better filtering applied in Non-Maximum Suppression (NMS).
- **Edge Detection Enhancement**: The bounding boxes are extended slightly to improve edge detection and ensure more accurate tracking of detected people.
- **NMS Parameters Tuning**: NMS parameters are adjusted to improve the handling of overlapping detections.
- **Frame Processing Frequency**: The frequency for processing frames is reduced to every 4th frame, optimizing processing time and performance.

## Code Explanation

### 1. **Imports and Dependencies**

The code uses the following libraries:

- `cv2` (OpenCV) for video processing and object detection.
- `numpy` for array manipulation.
- `json` for saving results in a JSON file.
- `pathlib` for working with file paths.
- `datetime` for handling timestamps.
- `os` for managing directories and saving images.

### 2. **SimpleCounter Class**

The `SimpleCounter` class is responsible for video processing, person detection, and counting.

#### `__init__` Method:

- Loads the YOLO model, sets the backend and target, and loads the COCO class labels.
- Creates an output directory to save frames (`_frames`).

#### `detect_people` Method:

- Detects people in a frame using YOLO. It converts the image into a blob and feeds it to the network.
- Filters detections for people based on a confidence score and applies bounding box adjustments for better detection.
- Uses Non-Maximum Suppression (NMS) to handle overlapping boxes.

#### `save_marked_frame` Method:

- Saves frames with drawn bounding boxes around detected people, confidence scores, and person IDs.
- Frames are saved in a specific directory (`_frames`) with filenames based on the interval and timestamp.

#### `count_people` Method:

- Processes the video, reading frames at regular intervals (every 4th frame in v3).
- Detects people and saves frames with the highest count per interval.
- Outputs results to a JSON file, including references to saved frames.

### 3. **Main Script Execution**

- The main script creates an instance of the `SimpleCounter` class and processes the video to detect and count people.
- The results are saved in a JSON file with the people count per interval, and a path to the saved frames is included.
- Optionally, the script can run in debug mode, showing frames with bounding boxes and confidence scores.

### 4. **Output**

The script generates:

- A **JSON file** containing the people count per interval and a reference to the verification frame.
- A **directory** (`video_frames`) containing the saved frames with bounding boxes for verification.

The JSON file includes:

- `interval`: Interval number.
- `timestamp`: Timestamp when the interval ended.
- `frame_number`: Frame number at the end of the interval.
- `people_count`: Number of detected people in the interval.
- `verification_frame`: File path to the saved frame.

### 5. **Debugging**

If `debug=True`, the program will display frames with bounding boxes, confidence values, and person IDs. You can press 'q' to close the debug window.

## Example Usage

```python
video_path = "video.mp4"
counter = SimpleCounter(video_path, interval_seconds=5, debug=True)
results = counter.count_people()
```

## Output:

- A JSON file (`video_counts.json`) containing the results.
- A folder (`video_frames`) containing saved frames for verification.

## Requirements

- Python 3.x
- OpenCV (`cv2` package)
- YOLOv4-Tiny weights and configuration files (`yolov4-tiny.weights`, `yolov4-tiny.cfg`)
- COCO class names file (`coco.names`)
- Test video can be found using link in the **Test Video link.txt** file.

---
