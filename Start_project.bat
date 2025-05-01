@echo off

start /min cmd /k "cd central_server && python -m uvicorn api:app --host 0.0.0.0 --port 8000 --reload"
start /min cmd /k "cd edge_server && python -m uvicorn api:app --host 0.0.0.0 --port 8001 --reload"
start /min cmd /k "cd desktop_app && streamlit run app.py"
start /min cmd /k "cd sensors/common_cameras && python camera.py"
start /min cmd /k "cd sensors/power_plants && python plants.py"
start /min cmd /k "cd sensors/traffic_cameras && python camera.py"
start /min cmd /k "cd sensors/weather_sensors && python weather_stream.py"
start /min cmd /k "cd edge_server/power_gen_pred && python realtime_pred.py"
start /min cmd /k "cd edge_server/weather_pred && python realtime_pred.py"
start http://localhost:5001
start http://localhost:8501
start http://localhost:3003