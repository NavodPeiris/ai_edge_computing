from fastapi import FastAPI, HTTPException, Request, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pathlib import Path
import tensorflow as tf
import numpy as np
import cv2
from keras.applications import VGG16
from keras.models import Model, Sequential
from weather_pred.weather_prediction import infer_multi_output

app = FastAPI()

# Load the trained model once at server startup
model = tf.keras.models.load_model('violence_detection/violence_detection_models/model.h5')

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    frames = []
    
    try:
        while True:
            # Receive frame data from the client
            data = await websocket.receive_bytes()
            
            # Decode the frame
            frame = np.frombuffer(data, dtype=np.uint8).reshape(224, 224, 3)
            frames.append(frame)
            
            # Process every 20 frames
            if len(frames) == 20:
                print("running inference...")
                # Prepare the frames for the model
                input_data = np.array(frames, dtype=np.float16) / 255.0  # Normalize

                image_model = VGG16(include_top=True, weights='imagenet')


                # We will use the output of the layer prior to the final
                # classification-layer which is named fc2. This is a fully-connected (or dense) layer.
                transfer_layer = image_model.get_layer('fc2')

                image_model_transfer = Model(inputs=image_model.input,
                                            outputs=transfer_layer.output)
                
                # Run inference
                transfer_values = image_model_transfer.predict(input_data, batch_size=len(frames))

                # Reshape to match the input shape of the custom model
                # Expected shape: (batch_size, 20, 4096)
                transfer_values_sequence = transfer_values.reshape(1, 20, 4096)

                prediction = model.predict(transfer_values_sequence)[0]
                violence_score = prediction[0]
                non_violence_score = prediction[1]
                if violence_score > non_violence_score:
                    print("violence detected")
                else:
                    print("no violence detected")
                
                # Clear the frames buffer
                frames = []
    except WebSocketDisconnect:
        print("Client disconnected")
    finally:
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
