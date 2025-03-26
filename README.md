#### start central server
```
cd central_server
uvicorn api:app --host 0.0.0.0 --port 8000 --reload
```

#### start edge server
```
cd edge_server
uvicorn api:app --host 0.0.0.0 --port 8001 --reload
```

#### start influxdb, grafana services
```
docker-compose up -d
```

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

#### run desktop-app:
```
cd desktop_app
streamlit run app.py
```