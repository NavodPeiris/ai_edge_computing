#### start central server
```
cd central_server
uvicorn api:app --host 0.0.0.0 --port 8000 --reload
```

#### start edge server1
```
cd edge_server1
uvicorn api:app --host 0.0.0.0 --port 8001 --reload
```

#### start influxdb, chronograf services
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