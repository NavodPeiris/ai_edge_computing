import asyncio
import json
import time
import traceback
import websockets
import cv2
import base64
import numpy as np
import random
from concurrent.futures import ThreadPoolExecutor

locations = ["malabe", "kandy", "mount lavinia", "maharagama"]

def send_video_sync(location):
    asyncio.run(send_video(location))
    

async def send_video(location):
    uri = "ws://localhost:8001/common_cam_ws"
    
    # Path to the video file
    video_path = "non_violence_videos/video.mp4"
    
    async with websockets.connect(uri) as websocket:
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"Error: Could not open video file {video_path}")
            return
        
        fps = cap.get(cv2.CAP_PROP_FPS)  # Get video FPS
        print(f"Detected FPS: {fps}")

        try:
            while True:
                start_time = time.time()  # Track frame time
                ret, frame = cap.read()
                
                # If the video ends, restart it (looping)
                if not ret:
                    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    continue
                
                # Encode frame as JPEG to reduce size
                _, buffer = cv2.imencode('.jpg', frame)
                encoded_frame = base64.b64encode(buffer).decode("utf-8")
                
                # Create JSON payload
                data = {
                    "location": location,
                    "frame": encoded_frame,
                    "fps": fps
                }
                
                # Send data to the server
                await websocket.send(json.dumps(data))

                # Maintain correct FPS
                elapsed_time = time.time() - start_time
                sleep_time = max(0, (1 / fps) - elapsed_time)
                await asyncio.sleep(sleep_time)
        finally:
            cap.release()


# Wrapper to catch and log exceptions in threads
def thread_wrapper(func, *args, **kwargs):
    try:
        func(*args, **kwargs)
    except Exception as e:
        print(f"Unhandled exception in thread: {e}")
        traceback.print_exc()  # Print the traceback for debugging


def main():
    try:
        with ThreadPoolExecutor(max_workers=len(locations)) as executor:
            # Store futures in a list
            futures = []

            # Stream station data
            for location in locations:
                futures.append(executor.submit(thread_wrapper, send_video_sync, location))

            # Wait for all futures to complete
            for future in futures:
                try:
                    future.result()  # Raise exceptions if any occurred in the thread
                except Exception as e:
                    print(f"Error in thread: {e}")
                    traceback.print_exc()  # Print traceback for debugging

    except KeyboardInterrupt:
        print("\nStreaming stopped.")


# Run the client
main()

