import time
import random
import traceback
from influxdb_client import InfluxDBClient, Point
from datetime import datetime, timedelta
import random
from power_gen_pred import infer_model
import pandas as pd
from concurrent.futures import ThreadPoolExecutor
from influxdb_client.client.write_api import SYNCHRONOUS

# InfluxDB connection parameters
url = "http://localhost:8086"  # InfluxDB 2.x URL
token = "3wvWUxmtdBM03hm9YgTEa91s6ofQ73G4gQ54uNR0Ek59zpJNMGOagj1UR1GKw3D1f5Elw-zS78rEwY7akZGmOw=="  # Authentication token
org = "fyp"      # Organization name
bucket = "solar_power_generation"  # Bucket name

# Initialize InfluxDB client
client = InfluxDBClient(url=url, token=token, org=org)

query_api = client.query_api()
write_api = client.write_api()

"""
# List of plant IDs
plant_ids = [
    "plant1", "plant2", "plant3", "plant4", "plant5", "plant6", "plant7"
]

plant_to_node = {
    "edge_server1": {
        "id": 1,
        "lat": 6.916528,
        "lon": 79.893854,
    },
    "plant1": {
        "id": 2,
        "lat": 6.924790,
        "lon": 79.881042,
    },
    "plant2": {
        "id": 3,
        "lat": 6.898374,
        "lon": 79.881699,
    },
    "plant3": {
        "id": 4,
        "lat": 6.880436,
        "lon": 79.882466,
    },
    "plant4": {
        "id": 5,
        "lat": 6.917724,
        "lon": 79.906338,
    },
    "plant5": {
        "id": 6,
        "lat": 6.896308,
        "lon": 79.908090,
    },
    "plant6": {
        "id": 7,
        "lat": 6.882285,
        "lon": 79.910718,
    },
    "plant7": {
        "id": 8,
        "lat": 6.881415,
        "lon": 79.894292,
    },
}
"""

# List of plant IDs
plant_ids = [
    "plant1", "plant2", "plant3"
]

plant_to_node = {
    "edge_server1": {
        "id": 1,
        "lat": 6.916528,
        "lon": 79.893854,
    },
    "plant1": {
        "id": 2,
        "lat": 6.924790,
        "lon": 79.881042,
    },
    "plant2": {
        "id": 3,
        "lat": 6.898374,
        "lon": 79.881699,
    },
    "plant3": {
        "id": 4,
        "lat": 6.880436,
        "lon": 79.882466,
    }
}

# Function to fetch the latest 4 points from InfluxDB for plant1
def fetch_latest_data(plant_id):
    
    query = f'''
    from(bucket: "{bucket}")
        |> range(start: -30d)  // Adjust the range as needed
        |> filter(fn: (r) => r._measurement == "solar_generation_data")
        |> filter(fn: (r) => r["plant_id"] == "{plant_id}")
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
                    "id": record["id"],
                    "title": record["title"],
                    "lat": record["lat"],
                    "lon": record["lon"],
                    "color": record["color"],
                    "plant_id": record["plant_id"],
                    "ambient_temperature": record["ambient_temperature"],
                    "module_temperature": record["module_temperature"],
                    "irradiation": record["irradiation"],
                    "period_generation": record["period_generation"],
                    "time": record["_time"]
                })

        points.sort(key=lambda x: x["time"], reverse=True)
        points = points[:4]
        #print(len(points))
        
        # If no points found, return empty DataFrame and None
        if not points:
            return pd.DataFrame(), None
        

        processed_points = []
        for point in points:
            processed_points.append({
                "AMBIENT_TEMPERATURE": point["ambient_temperature"],
                "MODULE_TEMPERATURE": point["module_temperature"],
                "IRRADIATION": point["irradiation"],
                "PERIOD_GENERATION": point["period_generation"]
            })
        
        # Convert the points to a pandas DataFrame
        latest_data = pd.DataFrame(processed_points)
        
        # Extract the time of the latest point
        latest_time = points[0]["time"] if not len(points) == 0 else None
        
        return latest_data, latest_time

    except Exception as e:
        print(f"Error fetching data: {e}")
        return pd.DataFrame(), None


def upcoming_cloud_cover_for_plant(plant_id):
    plant_node_id = plant_to_node[plant_id]["id"]
    try:
        is_cloud_coming = False
        estimated_arrival = 0
        current_next = plant_node_id  # Get the index for the target plant
        i = 0

        while True:
            # Flux query to find the next point in the chain
            query = f'''
            from(bucket: "{bucket}")
              |> range(start: -1d)  // Adjust the range as needed
              |> filter(fn: (r) => r._measurement == "sensor_grid_data")
              |> filter(fn: (r) => int(v: r.next) == {current_next})
              |> limit(n: 1)
            '''
            
            result = query_api.query(org=org, query=query)

            # Extract points
            points = []
            for table in result:
                for record in table.records:
                    points.append({
                        "id": record["id"],
                        "next": record["next"],
                        "irradiation": record["_value"],
                        "color": record["color"],
                        "time": record["_time"]
                    })

            points.sort(key=lambda x: x["time"], reverse=True)

            #print(f"points for {plant_id} : ", points)

            if not points:
                # Break if no point matches the current "next" value
                break

            i += 1
            point = points[0]

            # Check if cloud is coming (irradiation is 0)\
            #print(f"{plant_node_id} : {point['color']}")
            if point["color"] == "grey":
                is_cloud_coming = True
                estimated_arrival = i

            # Update the current_next to follow the chain
            current_next = point["id"]  # Assuming "id" is the unique identifier for the next point

        if is_cloud_coming:
            print("cloud coming to: ", plant_id)
        else:
            pass
            #print("no cloud coming to: ", plant_id)
        return is_cloud_coming, estimated_arrival

    except Exception as e:
        print(f"Error fetching chain for plant_id {plant_id}: {e}")
        return False, 0



# Function to write predicted data to InfluxDB
def write_predicted_data(prediction, latest_time, plant_id):
    # Add 1 minute to the latest time
    if latest_time:
        new_time = (latest_time + pd.Timedelta(minutes=1)).isoformat()
    else:
        new_time = datetime.utcnow().isoformat()  # Default to current UTC time if no latest_time provided

    query = f"""
    from(bucket: "{bucket}")
      |> range(start: {new_time}, stop: {new_time})
      |> filter(fn: (r) => r._measurement == "solar_generation_pred")
      |> filter(fn: (r) => r["plant_id"] == "{plant_id}")
      |> filter(fn: (r) => r["cloud_predicted_point"] == "true")
    """
    
    try:
        try:
            # Execute the query
            result = client.query_api().query(query)
            
            # Check if any points exist
            if result:
                print(f"A point already exists for plant_id {plant_id} at time {new_time} with cloud_predicted_point = True.")
                return  # Skip writing the new point
        except Exception as e:
            print(f"Error checking or writing data for plant_id {plant_id}: {e}")
            
        
        # If no conflict, write the new point
        pred_point = (
            Point("solar_generation_pred")  # Measurement name
            .tag("plant_id", str(plant_id))  # Plant ID from CSV
            .field("predicted_period_generation", prediction)
            .field("cloud_predicted_point", False)
            .time(new_time)  # Current timestamp in UTC
        )
        
        write_api.write(bucket=bucket, org=org, record=pred_point)  # Write the point to InfluxDB
        print(f"Predicted data written: {pred_point}")
    
    except Exception as e:
        print(f"Error checking or writing data for plant_id {plant_id}: {e}")


# Function to process data for a single plant
def process_plant(plant_id):
    while True:
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

        time.sleep(20)


def pred_cloud_coverage(plant_id):
    print("cloud pred running")
    while True:
        # Check if a cloud is coming for the plant
        is_cloud_coming, estimated_arrival = upcoming_cloud_cover_for_plant(plant_id)
        if is_cloud_coming:
            # Query the latest point for the plant
            query = f'''
            from(bucket: "{bucket}")
                |> range(start: -30d)  // Adjust the range as needed
                |> filter(fn: (r) => r._measurement == "solar_generation_data")
                |> filter(fn: (r) => r["plant_id"] == "{plant_id}")
                |> pivot(rowKey:["_time"], columnKey: ["_field"], valueColumn: "_value")  
                |> limit(n: 1)
            '''
            result = query_api.query(org=org, query=query)

            points = []
            for table in result:
                for record in table.records:
                    #print("record: ", record)
                    points.append({
                        "id": record["id"],
                        "title": record["title"],
                        "lat": record["lat"],
                        "lon": record["lon"],
                        "color": record["color"],
                        "plant_id": record["plant_id"],
                        "ambient_temperature": record["ambient_temperature"],
                        "module_temperature": record["module_temperature"],
                        "irradiation": record["irradiation"],
                        "period_generation": record["period_generation"],
                        "time": record["_time"]
                    })

            points.sort(key=lambda x: x["time"], reverse=True)

            if not points:
                print(f"No data found for plant_id: {plant_id}")

            point = points[0]

            new_time = (datetime.utcnow() + timedelta(minutes=estimated_arrival)).isoformat()

            if point["color"] != "red":
                # Update the color to yellow
                color_changed_point = (
                    Point("solar_generation_data")  # Measurement name
                    .tag("id", point["id"])  # Node Id
                    .tag("title", str(point["title"]))  # Title for node
                    .tag("lat", float(point["lat"]))  # Latitude of node
                    .tag("lon", float(point["lon"]))  # Longitude of node
                    .tag("color", "yellow")  # Color of node based on irradiation value
                    .tag("plant_id", str(point["plant_id"]))  # Plant ID
                    .field("ambient_temperature", float(point["ambient_temperature"]))  # Ambient temperature
                    .field("module_temperature", float(point["module_temperature"]))  # Module temperature
                    .field("irradiation", float(point["irradiation"]))  # Irradiation
                    .field("period_generation", float(point["period_generation"]))  # Daily yield
                    .time(datetime.utcnow())  # Original timestamp
                )

                write_api.write(bucket=bucket, org=org, record=color_changed_point)
                print(f"Color updated to yellow for plant_id: {plant_id}")

            # Create a prediction with 0 as cloud is coming
            cloud_pred_point = (
                Point("solar_generation_pred")  # Measurement name
                .tag("plant_id", str(plant_id))  # Plant ID
                .field("predicted_period_generation", 0.0)  # Predicted generation
                .field("cloud_predicted_point", True)  # Cloud prediction flag
                .time(new_time)  # Predicted time
            )

            write_api.write(bucket=bucket, org=org, record=cloud_pred_point)
            print(f"Cloud prediction written for plant_id: {plant_id}, time: {new_time}")
        
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
    print("realtime predictions for all plants... Press Ctrl+C to stop.")
    
    with ThreadPoolExecutor(max_workers=2*len(plant_ids)) as executor:
            # Store futures in a list
            futures = []

            # Stream plant data
            for plant_id in plant_ids:
                futures.append(executor.submit(thread_wrapper, process_plant, plant_id))
                futures.append(executor.submit(thread_wrapper, pred_cloud_coverage, plant_id))

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