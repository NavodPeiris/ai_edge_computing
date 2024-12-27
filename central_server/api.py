from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pathlib import Path

app = FastAPI()

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
    
    # Get list of files in the data folder of centralized server
    files = [file.name for file in DATA_FOLDER.iterdir() if file.is_file()]

    # Get edge server list
    edge_server_links = [
        "http://127.0.0.1:8001"     # edge server1
        "http://127.0.0.1:8002"     # edge server2
        "http://127.0.0.1:8003"     # edge server3
    ]

    return templates.TemplateResponse("index.html", {"request": request, "files": files, "edge_server_links": edge_server_links})

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
