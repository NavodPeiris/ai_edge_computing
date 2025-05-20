import time
import csv
from influxdb_client import InfluxDBClient, Point
from influxdb_client.client.write_api import SYNCHRONOUS
from datetime import datetime, time as dt_time, timedelta
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
import random
import traceback
import pandas as pd

# InfluxDB connection parameters
url = "http://localhost:8086"  # InfluxDB 2.x URL
token = "3wvWUxmtdBM03hm9YgTEa91s6ofQ73G4gQ54uNR0Ek59zpJNMGOagj1UR1GKw3D1f5Elw-zS78rEwY7akZGmOw=="  # Authentication token
org = "fyp"      # Organization name
bucket = "weather_data"  # Bucket name

# Initialize InfluxDB client
client = InfluxDBClient(url=url, token=token, org=org)
write_api = client.write_api(write_options=SYNCHRONOUS)
query_api = client.query_api()

#locations = ["malabe", "kandy", "mount lavinia", "maharagama"]
locations = ["malabe"]

# getting t-14 days
end_date = datetime.utcnow().date()
start_date = end_date - timedelta(days=14)

# Generate list of dates from start_date to end_date (inclusive)
date_list = [(start_date + timedelta(days=i)).isoformat() for i in range(15)]

print(date_list)

# Function to generate edge_server data for influxDB
def generate_edge_server_data(row, location):
    #print(row)
    return (
        Point("weather_data_gen")  # Measurement name
        .field("humidity", row["humidity"])  # Ambient temperature
        .field("rain", row["rain"])  # Module temperature
        .field("temperature", row["temperature"])  # Irradiation
        .tag("location", location)
        .time(row["datetime_utc"])  # Current timestamp in UTC
    )


# Function to stream data for a single plant with error handling and loop termination
def edge_server_stream():
    print(f"Starting streaming for edge_server")
    df = pd.read_csv("data/testset.csv")
    df.columns = df.columns.str.strip()  # Remove spaces in column names
    df.index = pd.to_datetime(df['datetime_utc'])  # Set datetime as index

    df = df.head(len(date_list))
    df['datetime_utc'] = date_list

    try:
        for index, row in df.iterrows():
            try:
                for location in locations:
                    point = generate_edge_server_data(row, location)
                    write_api = client.write_api()
                    write_api.write(bucket=bucket, org=org, record=point)  # Write the point to InfluxDB
                    print(f"Data written: {point}")
            except Exception as e:
                print(f"Error while writing data: {e}")
                traceback.print_exc()  # Print the traceback for debugging
                return  # Exit the loop and stop the thread
        time.sleep(60) # wait until data is written 
    except Exception as e:
        print(f"Error in stream: {e}")
        traceback.print_exc()  # Print the traceback for debugging
    finally:
        print(f"Stopping streaming")


# Wrapper to catch and log exceptions in threads
def thread_wrapper(func, *args, **kwargs):
    try:
        func(*args, **kwargs)
    except Exception as e:
        print(f"Unhandled exception in thread: {e}")
        traceback.print_exc()  # Print the traceback for debugging


# Main streaming function
def stream_data_from_csv():
    
    try:
        print("Streaming data from CSV to InfluxDB... Press Ctrl+C to stop.")
        with ThreadPoolExecutor(max_workers=1) as executor:
            futures = []
            futures.append(executor.submit(thread_wrapper, edge_server_stream))

            # Wait for all futures to complete
            for future in futures:
                try:
                    future.result()  # This will raise exceptions if any occurred in the thread
                except Exception as e:
                    print(f"Error in thread: {e}")
                    traceback.print_exc()  # Print the traceback for debugging
    except KeyboardInterrupt:
        print("\nStreaming stopped.")
    finally:
        client.close()

# Start streaming data
stream_data_from_csv()