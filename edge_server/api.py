import asyncio
import base64
import json
import shutil
import time
import concurrent
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


process_pool = concurrent.futures.ProcessPoolExecutor(max_workers=2)

# getting t-14 days
end_date = datetime.utcnow().date()
start_date = end_date - timedelta(days=14)

# Generate list of dates from start_date to end_date (inclusive)
common_cam_date_list = [(start_date + timedelta(days=i)).isoformat() for i in range(15)]
traffic_cam_date_list = [(start_date + timedelta(days=i)).isoformat() for i in range(15)]


@app.websocket("/common_cam_ws")
async def common_camera_websocket_endpoint(websocket: WebSocket):
    await websocket.accept()

    frame_queue = asyncio.Queue()
    recording_task = None
    location = "Unknown"

    try:
        while True:
            message = await websocket.receive_text()
            data = json.loads(message)

            if len(common_cam_date_list) > 0:

                location = data.get("location", "Unknown")
                fps = int(data.get("fps", 25))

                frame_data = base64.b64decode(data["frame"])
                frame_array = np.frombuffer(frame_data, dtype=np.uint8)
                frame = cv2.imdecode(frame_array, cv2.IMREAD_COLOR)
                

                if frame is not None:
                    await frame_queue.put((frame, fps, location))

                    if recording_task is None or recording_task.done():
                        recording_task = asyncio.create_task(
                            handle_people_stream(frame_queue) 
                        )

    except WebSocketDisconnect:
        print(f"Client from {location} disconnected")
    finally:
        if recording_task:
            recording_task.cancel()
        cv2.destroyAllWindows()

async def handle_people_stream(queue: asyncio.Queue):
    frames = []
    out = None
    fps = 25
    duration = 20
    location = "Unknown"
    video_filename = None
    fourcc = cv2.VideoWriter.fourcc(*"mp4v")

    while len(common_cam_date_list) > 0:
        try:
            frame, fps, location = await queue.get()
            if out is None:
                timestamp = datetime.utcnow().strftime('%Y%m%d_%H%M%S_%f')
                post_fix = f"{location}_{timestamp}"
                video_filename = f"people_counting_data/people_{post_fix}.avi"
                height, width, _ = frame.shape
                out = await asyncio.to_thread(cv2.VideoWriter, video_filename, fourcc, fps, (width, height))

            frames.append(frame)
            await asyncio.to_thread(out.write, frame)

            if len(frames) >= fps * duration:
                out.release()
                out = None

                results = await asyncio.get_event_loop().run_in_executor(
                    process_pool, count_people, video_filename, post_fix, 5
                )

                for i, res in enumerate(results):
                    point = (
                        Point("people_count")
                        .tag("location", location)
                        .field("count", res["people_count"])
                        .time(common_cam_date_list.pop(0))
                    )
                    write_api.write(bucket=common_camera_bucket, org=org, record=point)

                frames.clear()
                await asyncio.sleep(60)

        except asyncio.CancelledError:
            if out:
                out.release()
            break


@app.websocket("/traffic_cam_ws")
async def traffic_camera_websocket_endpoint(websocket: WebSocket):
    await websocket.accept()

    frame_queue = asyncio.Queue()
    recording_task = None
    location = "Unknown"

    try:
        while True:
            message = await websocket.receive_text()
            data = json.loads(message)

            if len(traffic_cam_date_list) > 0:

                location = data.get("location", "Unknown")
                fps = int(data.get("fps", 25))

                # Decode base64 frame
                frame_data = base64.b64decode(data["frame"])
                frame_array = np.frombuffer(frame_data, dtype=np.uint8)
                frame = cv2.imdecode(frame_array, cv2.IMREAD_COLOR)

                if frame is not None:
                    await frame_queue.put((frame, fps, location))

                    if recording_task is None or recording_task.done():
                        recording_task = asyncio.create_task(
                            handle_traffic_stream(frame_queue)
                        )

    except WebSocketDisconnect:
        print(f"Client from {location} disconnected")
    finally:
        if recording_task:
            recording_task.cancel()
        cv2.destroyAllWindows()


async def handle_traffic_stream(queue: asyncio.Queue):
    frames = []
    out = None
    fps = 25
    duration = 20
    location = "Unknown"
    video_filename = None
    fourcc = cv2.VideoWriter.fourcc(*"mp4v")

    while len(traffic_cam_date_list) > 0:
        try:
            frame, fps, location = await queue.get()

            if out is None:
                timestamp = datetime.utcnow().strftime('%Y%m%d_%H%M%S_%f')
                post_fix = f"{location}_{timestamp}"
                video_filename = f"traffic_counting_data/traffic_{post_fix}.avi"
                height, width, _ = frame.shape
                out = await asyncio.to_thread(cv2.VideoWriter, video_filename, fourcc, fps, (width, height))

            frames.append(frame)
            await asyncio.to_thread(out.write, frame)

            if len(frames) >= fps * duration:
                out.release()
                out = None

                # Run inference in separate process
                counting_res = await asyncio.get_event_loop().run_in_executor(
                    process_pool, count_vehicles, video_filename, post_fix
                )

                try:
                    vehicles_in = sum(counting_res['vehicles_in'].values())
                    vehicles_out = sum(counting_res['vehicles_out'].values())

                    pred_point = (
                        Point("vehicles_count")
                        .tag("location", location)
                        .field("vehicles_coming_in", vehicles_in)
                        .field("vehicles_going_out", vehicles_out)
                        .time(traffic_cam_date_list.pop(0))
                    )

                    write_api.write(bucket=traffic_camera_bucket, org=org, record=pred_point)
                    print(f"Traffic data written: {pred_point}")

                except Exception as e:
                    print(f"Error during data processing or writing to DB: {e}")

                frames.clear()
                await asyncio.sleep(60)

        except asyncio.CancelledError:
            if out:
                out.release()
            break
        

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
