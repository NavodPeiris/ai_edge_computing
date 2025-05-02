import base64
import json
import shutil
import time
from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pathlib import Path
import tensorflow as tf
import numpy as np
import cv2
from keras.applications import VGG16
from keras.models import Model, Sequential
from classes.flwr_server_req import FlwrRequestParams, FlwrStopParams, DeliverModel, DeliverScaler
import subprocess
import os
import glob

import traceback
from influxdb_client import InfluxDBClient, Point
from datetime import datetime, timedelta
import random
from datetime import datetime, timezone

import pandas as pd
from concurrent.futures import ThreadPoolExecutor
from influxdb_client.client.write_api import SYNCHRONOUS
from vehicle_counter import count_vehicles
from people_counter import count_people


# InfluxDB connection parameters
url = "http://localhost:8086"  # InfluxDB 2.x URL
token = "3wvWUxmtdBM03hm9YgTEa91s6ofQ73G4gQ54uNR0Ek59zpJNMGOagj1UR1GKw3D1f5Elw-zS78rEwY7akZGmOw=="  # Authentication token
org = "fyp"      # Organization name
common_camera_bucket = "common_camera_data"  # Bucket name
traffic_camera_bucket = "traffic_camera_data"  # Bucket name

# Initialize InfluxDB client
client = InfluxDBClient(url=url, token=token, org=org)

query_api = client.query_api()
write_api = client.write_api()


app = FastAPI()

# Load the trained model once at server startup
# model = tf.keras.models.load_model('violence_detection/violence_detection_models/model.h5')

try:
    model = tf.keras.models.load_model('violence_detection/violence_detection_models/model.h5')
except FileNotFoundError:
    print("Warning: Model file not found. Some functionality will be limited.")
    model = None

import psutil

def terminate_process(pid):
    try:
        # Find the process by PID
        process = psutil.Process(pid)
        
        # Terminate the process
        process.terminate()
        
        # Wait for the process to terminate
        process.wait()
        
        print(f"Process with PID {pid} terminated successfully.")
    except psutil.NoSuchProcess:
        print(f"No process found with PID {pid}.")
    except psutil.AccessDenied:
        print(f"Access denied to terminate the process with PID {pid}.")
    except psutil.ZombieProcess:
        print(f"Process with PID {pid} is a zombie process.")
    except Exception as e:
        print(f"Error terminating process: {str(e)}")


def clean_up_models(folder_path):
    # Get all .h5 files in the folder
    model_files = glob.glob(os.path.join(folder_path, "model_round_*.h5"))
    
    # If no model files are found, return
    if not model_files:
        print("No model files found.")
        return

    # Sort the model files by round number (extracting the number from the filename)
    model_files.sort(key=lambda x: int(x.split('_')[-1].split('.')[0]), reverse=True)

    # The latest model (the first in the sorted list)
    latest_model = model_files[0]

    # Delete all models except the latest one
    for model_file in model_files[1:]:
        os.remove(model_file)
        print(f"Deleted {model_file}")

    # Rename the latest model to model.h5
    new_model_name = os.path.join(folder_path, "model.h5")
    # Check if the new model already exists
    if os.path.exists(new_model_name):
        # If it exists, remove it (this will delete the existing model.h5)
        os.remove(new_model_name)
        print(f"Deleted existing model: {new_model_name}")

    # Rename the latest model to model.h5
    os.rename(latest_model, new_model_name)
    print(f"Renamed {latest_model} to {new_model_name}")


# Load the base model for feature extraction
image_model = VGG16(include_top=True, weights='imagenet')
transfer_layer = image_model.get_layer('fc2')
image_model_transfer = Model(inputs=image_model.input, outputs=transfer_layer.output)

@app.websocket("/common_cam_ws")
async def common_camera_websocket_endpoint(websocket: WebSocket):
    print("WebSocket Connection Attempt")
    await websocket.accept()
    violence_detection_frames = []
    people_counting_frames = []

    fourcc = cv2.VideoWriter.fourcc(*"mp4v")
    out = None  # Video writer (initialized later)
    video_filename = f"people_counting_data/people_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}.avi"

    try:
        while True:
            # Receive JSON message from the client
            message = await websocket.receive_text()
            data = json.loads(message)

            # Extract location
            location = data.get("location", "Unknown")
            # Extract location
            fps = data.get("fps", 25)

            duration = 20  # Record duration in seconds
            max_frames = int(fps * duration)  # Total frames for 20 seconds

            # Decode the Base64-encoded frame
            frame_data = base64.b64decode(data["frame"])
            frame_array = np.frombuffer(frame_data, dtype=np.uint8)
            frame = cv2.imdecode(frame_array, cv2.IMREAD_COLOR)

            if frame is None:
                print("Error: Failed to decode frame")
                continue

            height, width, _ = frame.shape  # Get frame dimensions

            # Initialize video writer once
            if out is None:
                out = cv2.VideoWriter(video_filename, fourcc, fps, (width, height))

            people_counting_frames.append(frame)
            out.write(frame)  # Write frame to video

            # Resize frame to 224x224 (to match model input size)
            violence_detection_frame = cv2.resize(frame, (224, 224))
            violence_detection_frames.append(violence_detection_frame)

            # Process every 20 frames
            if len(violence_detection_frames) == 20:
                print(f"Running violence detection... (Location: {location})")

                # Normalize frames
                input_data = np.array(violence_detection_frames, dtype=np.float16) / 255.0

                # Run feature extraction
                transfer_values = image_model_transfer.predict(input_data, batch_size=len(violence_detection_frames))

                # Reshape for the custom model
                transfer_values_sequence = transfer_values.reshape(1, 20, 4096)

                # Run inference with the custom model
                prediction = model.predict(transfer_values_sequence)[0]
                violence_score, non_violence_score = prediction[0], prediction[1]

                if violence_score > non_violence_score:
                    print(f"Violence detected at {location}")
                    
                    try:
                        # If no conflict, write the new point
                        pred_point = (
                            Point("violence")  # Measurement name
                            .tag("location", location) 
                            .field("is_violence", int(1))
                            .time(datetime.utcnow())  # Current timestamp in UTC
                        )
                        
                        write_api.write(bucket=common_camera_bucket, org=org, record=pred_point)  # Write the point to InfluxDB
                        print(f"violence data written: {pred_point}")
                    
                    except Exception as e:
                        print(f"Error checking or writing data: {e}")
                    
                else:
                    print(f"No violence detected at {location}")
                    try:
                        # If no conflict, write the new point
                        pred_point = (
                            Point("violence")  # Measurement name
                            .tag("location", location) 
                            .field("is_violence", int(0))
                            .time(datetime.utcnow())  # Current timestamp in UTC
                        )
                        
                        write_api.write(bucket=common_camera_bucket, org=org, record=pred_point)  # Write the point to InfluxDB
                        print(f"violence data written: {pred_point}")
                    
                    except Exception as e:
                        print(f"Error checking or writing data: {e}")

                # Clear the frame buffer
                violence_detection_frames = []


            # Stop recording after 20 seconds
            if len(people_counting_frames) == max_frames:
                print(f"Video saved: {video_filename}")

                # Reset for the next 20-sec segment
                people_counting_frames = []
                out.release()

                interval_seconds = 5
                results = count_people(video_filename, interval_seconds)

                for i in range(len(results)):
                    try:
                        # If no conflict, write the new point
                        pred_point = (
                            Point("people_count")  # Measurement name
                            .tag("location", location) 
                            .field("count", results[i]["people_count"])
                            .time(datetime.utcnow() + pd.Timedelta(seconds=interval_seconds * i))  # Current timestamp in UTC
                        )
                        
                        write_api.write(bucket=common_camera_bucket, org=org, record=pred_point)  # Write the point to InfluxDB
                        print(f"people count data written: {pred_point}")
                    
                    except Exception as e:
                        print(f"Error checking or writing data: {e}")
    except WebSocketDisconnect:
        print("Client disconnected")
    finally:
        cv2.destroyAllWindows()


@app.websocket("/traffic_cam_ws")
async def traffic_camera_websocket_endpoint(websocket: WebSocket):
    print("WebSocket Connection Attempt")
    await websocket.accept()
    
    frames = []
    
    fourcc = cv2.VideoWriter.fourcc(*"mp4v")
    out = None  # Video writer (initialized later)
    video_filename = f"traffic_counting_data/traffic_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}.avi"

    try:
        while True:
            # Receive JSON message from the client
            message = await websocket.receive_text()
            data = json.loads(message)

            # Extract location
            location = data.get("location", "Unknown")
            # Extract location
            fps = data.get("fps", 25)

            duration = 20  # Record duration in seconds
            max_frames = int(fps * duration)  # Total frames for 20 seconds

            # Decode the Base64-encoded frame
            frame_data = base64.b64decode(data["frame"])
            frame_array = np.frombuffer(frame_data, dtype=np.uint8)
            frame = cv2.imdecode(frame_array, cv2.IMREAD_COLOR)

            if frame is None:
                print("Error: Failed to decode frame")
                continue

            height, width, _ = frame.shape  # Get frame dimensions

            # Initialize video writer once
            if out is None:
                out = cv2.VideoWriter(video_filename, fourcc, fps, (width, height))

            frames.append(frame)
            out.write(frame)  # Write frame to video

            # Stop recording after 20 seconds
            if len(frames) == max_frames:
                print(f"Video saved: {video_filename}")

                # Reset for the next 20-sec segment
                frames = []
                out.release()

                count_vehicles(video_filename)

                counting_res = json.load(open('counting_results.json', 'r'))
                vehicles_in = 0
                vehicles_out = 0

                for line_idx in counting_res['vehicles_in'].keys():
                    vehicles_in = counting_res['vehicles_in'][line_idx]
                    vehicles_out = counting_res['vehicles_out'][line_idx]

                try:
                    # If no conflict, write the new point
                    pred_point = (
                        Point("vehicles_count")  # Measurement name
                        .tag("location", location) 
                        .field("vehicles_coming_in", vehicles_in)
                        .field("vehicles_going_out", vehicles_out)
                        .time(datetime.utcnow())  # Current timestamp in UTC
                    )
                    
                    write_api.write(bucket=traffic_camera_bucket, org=org, record=pred_point)  # Write the point to InfluxDB
                    print(f"traffic data written: {pred_point}")
                
                except Exception as e:
                    print(f"Error checking or writing data: {e}")

    except WebSocketDisconnect:
        print("Client disconnected")
    finally:
        if out:
            out.release()
        cv2.destroyAllWindows()


# Path to the data folder
DATA_FOLDER = Path("./storage")

# Mount templates and static files
templates = Jinja2Templates(directory="templates")

# Route to serve the web page
@app.get("/", response_class=HTMLResponse)
def list_files(request: Request):
    """
    Render the file list on a web page.
    """
    if not DATA_FOLDER.exists():
        raise HTTPException(status_code=404, detail="Data folder not found.")
    
    # Get list of files in the data folder
    files = [file.name for file in DATA_FOLDER.iterdir() if file.is_file()]
    return templates.TemplateResponse("index.html", {"request": request, "files": files})

# Route to handle file downloads
@app.get("/download/{file_name}")
def download_file(file_name: str):
    """
    Download a file from the data folder.
    """
    file_path = DATA_FOLDER / file_name
    if not file_path.exists() or not file_path.is_file():
        raise HTTPException(status_code=404, detail="File not found.")
    
    return FileResponse(file_path, media_type="application/octet-stream", filename=file_name)

@app.post("/start_flwr_server/")
async def start_process(params: FlwrRequestParams):
    
    # Prepare the command to run the subprocess
    command = [
        "python",
        "flwr_server.py",  # Replace with your actual script path
        "--rounds", str(params.rounds),
        "--model_json", str(params.model_json),
        "--save_path", str(params.save_path)
    ]
    
    try:
        # Start the external Python script as a subprocess
        process = subprocess.Popen(command)
        print(f"Started subprocess with PID: {process.pid}")

        # Return the PID of the subprocess
        return {"pid": process.pid}
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error starting process: {str(e)}")


@app.post("/stop_flwr_server/")
async def stop_process(params: FlwrStopParams):
    try:
        terminate_process(params.pid)
        base_path = "/".join(params.save_path.split("/")[:-1])
        print(base_path)
        clean_up_models(base_path)

        # Return the PID of the subprocess
        return {"message": "successfully terminated"}
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error terminating process: {str(e)}")
    

@app.post("/upload_scaler/")
async def upload_scaler(save_path: str = Form(...), file: UploadFile = File(...)):
    # Ensure the uploaded file is a .pkl file
    if not file.filename.endswith(".pkl"):
        raise HTTPException(status_code=400, detail="Only .pkl files are allowed")
    
    base_path = "/".join(save_path.split("/")[:-1])
    os.makedirs(base_path, exist_ok=True)
    
    # Save the uploaded file
    with open(save_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    return {"message": "Scaler uploaded successfully", "filename": file.filename, "saved_path": save_path}


@app.post("/deliver_model/")
async def deliver_model(params: DeliverModel):
    if not os.path.exists(params.model_path):
        raise HTTPException(status_code=404, detail="Model file not found")
    
    return FileResponse(params.model_path, filename=os.path.basename(params.model_path))

@app.post("/deliver_scaler/")
async def deliver_model(params: DeliverScaler):
    if not os.path.exists(params.scaler_path):
        raise HTTPException(status_code=404, detail="Scaler file not found")
    
    return FileResponse(params.scaler_path, filename=os.path.basename(params.scaler_path))
