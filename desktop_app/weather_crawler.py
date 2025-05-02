import time
import random
import traceback
from influxdb_client import InfluxDBClient, Point
from datetime import datetime, timedelta
import random
import pandas as pd
from concurrent.futures import ThreadPoolExecutor
from influxdb_client.client.write_api import SYNCHRONOUS

# InfluxDB connection parameters
url = "http://localhost:8086"  # InfluxDB 2.x URL
token = "3wvWUxmtdBM03hm9YgTEa91s6ofQ73G4gQ54uNR0Ek59zpJNMGOagj1UR1GKw3D1f5Elw-zS78rEwY7akZGmOw=="  # Authentication token
org = "fyp"      # Organization name
bucket = "weather_data"  # Bucket name

# Initialize InfluxDB client
client = InfluxDBClient(url=url, token=token, org=org)

query_api = client.query_api()
write_api = client.write_api()


def fetch_latest_data():
    
    query = f'''
    from(bucket: "{bucket}")
        |> range(start: -30d)  // Adjust the range as needed
        |> filter(fn: (r) => r._measurement == "weather_data_gen")
        |> pivot(rowKey:["_time"], columnKey: ["_field"], valueColumn: "_value")  
    '''

    try:
        # Execute the query
        result = query_api.query(org=org, query=query)

        # Extract the points into a list of dictionaries
        points = []

        for table in result:
            for record in table.records:
                points.append({
                    "humidity": record["humidity"],
                    "rain": record["rain"],
                    "temperature": record["temperature"],
                    "location": record["location"],
                    "time": record["_time"]
                })

        points.sort(key=lambda x: x["time"], reverse=True)

        # If no points found, return empty DataFrame and None
        if not points:
            return pd.DataFrame(), None
        

        processed_points = []
        for point in points:
            point_date = pd.to_datetime(point["time"]).strftime('%Y-%m-%d')
            processed_points.append({
                "date": point_date,
                "location": point["location"],
                "humidity": point["humidity"],
                "rain": point["rain"],
                "temperature": point["temperature"]
            })
        
        # Convert the points to a pandas DataFrame
        latest_data = pd.DataFrame(processed_points)

        # Group by 'date' and calculate the average of the metrics
        grouped_data = latest_data.groupby(['location', 'date']).agg({
            'humidity': 'mean',
            'rain': 'mean',
            'temperature': 'mean'
        }).reset_index()

        grouped_data['rain'] = grouped_data['rain'].apply(lambda x: 0 if x < 0.5 else 1)

        print(grouped_data)
        
        return grouped_data

    except Exception as e:
        print(f"Error fetching data: {e}")
        return pd.DataFrame()

'''
fetch_latest_data()
'''
