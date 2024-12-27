import time
import random
from influxdb import InfluxDBClient
from datetime import datetime
import random
from concurrent.futures import ThreadPoolExecutor

# InfluxDB connection parameters
host = "localhost"  # InfluxDB host
port = 8086         # InfluxDB port
username = "navod"  # Username for InfluxDB
password = "1234"   # Password for InfluxDB
database = "power_data"  # Database name

# Initialize InfluxDB client
client = InfluxDBClient(host=host, port=port, username=username, password=password, database=database)

# Function to generate random data
def generate_random_data(plant_id):
    return {
        "measurement": "solar_data",  # Measurement name
        "tags": {
            "plant_id": plant_id,  # Random plant ID
        },
        "fields": {
            "ambient_temperature": round(random.uniform(21.0, 35.0), 2),  # Random temperature
            "module_temperature": round(random.uniform(18.0, 58.0), 2),  # Random module temperature
            "irradiation": round(random.uniform(0.0, 1.0), 2),  # Random irradiation
            "daily_yield": round(random.uniform(0.0, 8300.0), 2),  # Random daily yield
        },
        "time": datetime.utcnow().isoformat()  # Current timestamp in UTC
    }

def plant_stream(plant_id):
    point = generate_random_data(plant_id)
    client.write_points([point])  # Write the point to InfluxDB
    print(f"Data written: {point}")

# Stream random data continuously
try:
    print("Streaming random data to InfluxDB... Press Ctrl+C to stop.")
    while True:
        
        # List of plant IDs
        plant_ids = [
            "plant1", "plant2", "plant3", "plant4", "plant5", "plant6", "plant7"
        ]

        # Process predictions for all plants in parallel
        with ThreadPoolExecutor(max_workers=7) as executor:
            executor.map(plant_stream, plant_ids)

        time.sleep(60)  # Wait for 60 seconds before sending the next data point
except KeyboardInterrupt:
    print("\nStreaming stopped.")
finally:
    client.close()
