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