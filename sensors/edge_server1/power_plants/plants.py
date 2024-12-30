import time
import csv
from influxdb import InfluxDBClient
from datetime import datetime
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor

# InfluxDB connection parameters
host = "localhost"  # InfluxDB host
port = 8086         # InfluxDB port
username = "navod"  # Username for InfluxDB
password = "1234"   # Password for InfluxDB
database = "solar_power_generation"  # Database name

# Initialize InfluxDB client
client = InfluxDBClient(host=host, port=port, username=username, password=password, database=database)

# Function to parse data from CSV and prepare for InfluxDB
def parse_csv_row_to_point(row):
    return {
        "measurement": "solar_generation_data",  # Measurement name
        "tags": {
            "plant_id": row["PLANT_ID"],  # Plant ID from CSV
        },
        "fields": {
            "ambient_temperature": float(row["AMBIENT_TEMPERATURE"]),  # Ambient temperature
            "module_temperature": float(row["MODULE_TEMPERATURE"]),  # Module temperature
            "irradiation": float(row["IRRADIATION"]),  # Irradiation
            "daily_yield": float(row["DAILY_YIELD"]),  # Daily yield
        },
        "time": datetime.utcnow().isoformat()  # Current timestamp in UTC
    }

# Function to stream data for a single plant
def plant_stream(plant_data):
    plant_id, rows = plant_data
    print(f"Starting streaming for {plant_id}")
    while True:
        for row in rows:
            point = parse_csv_row_to_point(row)
            client.write_points([point])  # Write the point to InfluxDB
            print(f"Data written for {plant_id}: {point}")
            time.sleep(60)  # Wait for 60 seconds before processing the next row

# Load CSV data into memory and group by PLANT_ID
def load_and_group_csv_data(file_path):
    with open(file_path, mode="r") as file:
        reader = csv.DictReader(file)
        grouped_data = defaultdict(list)
        for row in reader:
            grouped_data[row["PLANT_ID"]].append(row)
    return grouped_data

# Main streaming function
def stream_data_from_csv(file_path):
    grouped_data = load_and_group_csv_data(file_path)
    try:
        print("Streaming data from CSV to InfluxDB... Press Ctrl+C to stop.")
        with ThreadPoolExecutor(max_workers=len(grouped_data)) as executor:
            executor.map(plant_stream, grouped_data.items())
    except KeyboardInterrupt:
        print("\nStreaming stopped.")
    finally:
        client.close()

# Path to the CSV file
csv_file_path = "data/part_1.csv"

# Start streaming data
stream_data_from_csv(csv_file_path)
