import time
import random
import traceback
from influxdb_client import InfluxDBClient, Point
from datetime import datetime, timedelta
import random
from weather_model_infer import infer_multi_output
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

locations = ["malabe", "kandy", "mount lavinia", "maharagama"]

# Function to fetch the latest 7 points from InfluxDB
def fetch_latest_data(location):
    
    query = f'''
    from(bucket: "{bucket}")
        |> range(start: -30d)  // Adjust the range as needed
        |> filter(fn: (r) => r._measurement == "weather_data_gen")
        |> filter(fn: (r) => r.location == "{location}")  // Filter by location tag
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
                    "time": record["_time"]
                })

        points.sort(key=lambda x: x["time"], reverse=True)
        points = points[:7]

        # If no points found, return empty DataFrame and None
        if not points:
            return pd.DataFrame(), None
        

        processed_points = []
        for point in points:
            processed_points.append({
                "humidity": point["humidity"],
                "rain": point["rain"],
                "temperature": point["temperature"]
            })
        
        # Convert the points to a pandas DataFrame
        latest_data = pd.DataFrame(processed_points)
        
        # Extract the time of the latest point
        latest_time = points[0]["time"] if not len(points) == 0 else None
        
        return latest_data, latest_time

    except Exception as e:
        print(f"Error fetching data: {e}")
        return pd.DataFrame(), None


# Function to write predicted data to InfluxDB
def write_predicted_data(prediction, latest_time, location):
    
    # Add 1 minute to the latest time
    if latest_time:
        new_time = (latest_time + pd.Timedelta(minutes=1)).isoformat()
    else:
        new_time = datetime.utcnow().isoformat()  # Default to current UTC time if no latest_time provided

    try:
        # If no conflict, write the new point
        pred_point = (
            Point("weather_data_pred")  # Measurement name
            .field("humidity", prediction[0])  # Ambient temperature
            .field("rain", int(prediction[1] >= 0.7))  # Module temperature
            .field("temperature", prediction[2])  # Irradiation
            .tag("location", location)
            .time(new_time)  # Current timestamp in UTC
        )
        
        write_api.write(bucket=bucket, org=org, record=pred_point)  # Write the point to InfluxDB
        print(f"Predicted data written: {pred_point}")
    
    except Exception as e:
        print(f"Error checking or writing data: {e}")


# Function to process data
def process_infer():
    while True:
        for location in locations:
            try:
                # Fetch the latest 7 points
                latest_data, latest_time = fetch_latest_data(location)
                
                if len(latest_data) == 7:
                    # Predict the next point using the infer_model function
                    prediction = infer_multi_output(latest_data)
                    
                    # Write the predicted data to InfluxDB
                    write_predicted_data(prediction, latest_time, location)
                    print(f"Prediction written successfully.")
                else:
                    print(f"Not enough data to make a prediction.")
            except Exception as e:
                print(f"Error processing: {e}")

        time.sleep(20)


# Wrapper to catch and log exceptions in threads
def thread_wrapper(func, *args, **kwargs):
    try:
        func(*args, **kwargs)
    except Exception as e:
        print(f"Unhandled exception in thread: {e}")
        traceback.print_exc()  # Print the traceback for debugging


# Stream random data continuously and infer next point
try:
    print("realtime predictions for weather... Press Ctrl+C to stop.")
    
    with ThreadPoolExecutor(max_workers=1) as executor:
            # Store futures in a list
            futures = []

            futures.append(executor.submit(thread_wrapper, process_infer))
                
            # Wait for all futures to complete
            for future in futures:
                try:
                    future.result()  # Raise exceptions if any occurred in the thread
                except Exception as e:
                    print(f"Error in thread: {e}")
                    traceback.print_exc()  # Print traceback for debugging


except KeyboardInterrupt:
    print("\nStreaming stopped.")
finally:
    client.close()