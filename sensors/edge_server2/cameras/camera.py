import cv2
import asyncio
import websockets
import numpy as np

async def send_video():
    uri = "ws://localhost:8001/ws"
    
    # Path to the video file
    video_path = "non_violence_videos/v1.mpg"
    
    async with websockets.connect(uri) as websocket:
        # Open the video file
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"Error: Could not open video file {video_path}")
            return
        
        try:
            while True:
                # Read a frame from the video
                ret, frame = cap.read()
                
                # If the video ends, restart it (looping)
                if not ret:
                    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    continue
                
                # Resize the frame to 224x224
                frame = cv2.resize(frame, (224, 224))
                
                # Send the frame to the server
                await websocket.send(frame.tobytes())
        finally:
            cap.release()

# Run the client
asyncio.run(send_video())
