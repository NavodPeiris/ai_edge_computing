<p align="center">
  <img src="architecture.png" />
</p>

### Useful Info

#### grafana

```
username: admin
pword: fyp12345
```

#### influxdb username: navod

```
username: navod
pword: fyp12345
organization: fyp
token: 3wvWUxmtdBM03hm9YgTEa91s6ofQ73G4gQ54uNR0Ek59zpJNMGOagj1UR1GKw3D1f5Elw-zS78rEwY7akZGmOw==
```

#### Access MLflow ui

```
url:http://localhost:5001/
```

### Run The Simulation Manually

#### 1. start influxdb, grafana, MLflow, MySQL services
```
docker-compose up -d
```

#### 2. ingest models to MLflow

run temp python files in mlflow folder

#### 3. start central server

```
cd central_server
uvicorn api:app --host 0.0.0.0 --port 8000 --reload
```

#### 4. ingest events

```
cd central_server/event_ingester
python events_ingest.py
```

#### 5. start edge server

```
cd edge_server
uvicorn api:app --host 0.0.0.0 --port 8001 --reload
```

#### 6. start common_cameras

```
cd sensors/common_cameras
python camera.py
```

#### 7. start power plants

```
cd sensors/power_plants
python plants.py
```

#### 8. start traffic cameras

```
cd sensors/traffic_cameras
python camera.py
```

#### 9. start weather data stream

```
cd sensors/weather_sensors
python weather_stream.py
```

#### 10. start realtime solar power prediction
```
cd edge_server/power_gen_pred
python realtime_pred.py
```

#### 11. start realtime weather prediction
```
cd edge_server/weather_pred
python realtime_pred.py
```

#### 12. start realtime traffic prediction
```
cd edge_server/traffic_pred
python traffic_pred.py
```

#### 13. start realtime crowd prediction
```
cd edge_server/crowd_pred
python crowd_pred.py
```

#### 14. run local web app:

```
cd desktop_app
streamlit run app.py
```

login creds:
```
username: navod
password: navod123
```


### Run The Simulation Automatically (This is faulty! and make system slower!)

- Use **Start_project.bat**
- make sure docker is running
- run command:
```
start Start_project.bat
```

