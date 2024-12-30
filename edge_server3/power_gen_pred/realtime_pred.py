import time
import random
from influxdb import InfluxDBClient
from datetime import datetime
import random
from power_gen_pred import infer_model
import pandas as pd
from concurrent.futures import ThreadPoolExecutor

# InfluxDB connection parameters
host = "localhost"  # InfluxDB host
port = 8086         # InfluxDB port
username = "navod"  # Username for InfluxDB
password = "1234"   # Password for InfluxDB
database = "solar_power_generation"  # Database name

# Initialize InfluxDB client
client = InfluxDBClient(host=host, port=port, username=username, password=password, database=database)


# Function to fetch the latest 4 points from InfluxDB for plant1
def fetch_latest_data(plant_id):
    query = f"""
    SELECT 
        "ambient_temperature" AS "AMBIENT_TEMPERATURE", 
        "module_temperature" AS "MODULE_TEMPERATURE", 
        "irradiation" AS "IRRADIATION", 
        "daily_yield" AS "DAILY_YIELD"
    FROM "solar_generation_data" 
    WHERE "plant_id" = '{plant_id}' 
    ORDER BY time DESC 
    LIMIT 4
    """
    result = client.query(query)
    points = list(result.get_points())
    
    if not points:
        return pd.DataFrame(), None  # Return empty DataFrame and None if no data found

    # Convert the list of points to a pandas DataFrame
    latest_data = pd.DataFrame(points)
    
    # Extract the time of the latest point
    latest_time = pd.to_datetime(latest_data['time'].iloc[0]) if not latest_data.empty else None
    
    return latest_data, latest_time


# Function to write predicted data to InfluxDB
def write_predicted_data(prediction, latest_time, plant_id):
    # Add 1 minute to the latest time
    if latest_time:
        new_time = (latest_time + pd.Timedelta(minutes=1)).isoformat()
    else:
        new_time = datetime.utcnow().isoformat()  # Default to current UTC time if no latest_time provided

    point = {
        "measurement": "solar_generation_pred",  # Measurement name
        "tags": {
            "plant_id": plant_id,  # Random plant ID
        },
        "fields": {
            "predicted_daily_yield": prediction,  # Predicted daily yield
        },
        "time": new_time # Current timestamp in UTC
    }
    client.write_points([point])
    print(f"Predicted data written: {point}")


# Function to process data for a single plant
def process_plant(plant_id):
    try:
        # Fetch the latest 4 points for the plant
        latest_data, latest_time = fetch_latest_data(plant_id)
        
        if len(latest_data) == 4:
            # Predict the next point using the infer_model function
            prediction = infer_model(latest_data)
            
            # Write the predicted data to InfluxDB
            write_predicted_data(prediction, latest_time, plant_id)
            print(f"Prediction for {plant_id} written successfully.")
        else:
            print(f"Not enough data for {plant_id} to make a prediction.")
    except Exception as e:
        print(f"Error processing plant {plant_id}: {e}")


# Stream random data continuously and infer next point
try:
    print("realtime predictions for all plants... Press Ctrl+C to stop.")
    while True:
        # List of plant IDs
        plant_ids = [
            "plant15", "plant16", "plant17", "plant18", "plant19", "plant20", "plant21"
        ]

        # Process predictions for all plants in parallel
        with ThreadPoolExecutor(max_workers=7) as executor:
            executor.map(process_plant, plant_ids)
        
        # Wait for 60 seconds before sending the next data point
        time.sleep(60)


except KeyboardInterrupt:
    print("\nStreaming stopped.")
finally:
    client.close()