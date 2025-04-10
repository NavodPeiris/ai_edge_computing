@echo off

start /min cmd /k "cd central_server && python -m uvicorn api:app --host 0.0.0.0 --port 8000 --reload"
start /min cmd /k "cd edge_server && python -m uvicorn api:app --host 0.0.0.0 --port 8001 --reload"
start /min cmd /k "cd desktop_app && streamlit run app.py"
start http://localhost:5001
start http://localhost:8086
start http://localhost:3003