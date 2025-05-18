import time
import csv
from influxdb_client import InfluxDBClient, Point
from influxdb_client.client.write_api import SYNCHRONOUS
from datetime import datetime, time as dt_time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
import random
import traceback
from station_gen import station_provider

# InfluxDB connection parameters
url = "http://localhost:8086"  # InfluxDB 2.x URL
token = "3wvWUxmtdBM03hm9YgTEa91s6ofQ73G4gQ54uNR0Ek59zpJNMGOagj1UR1GKw3D1f5Elw-zS78rEwY7akZGmOw=="  # Authentication token
org = "fyp"      # Organization name
bucket = "solar_power_generation"  # Bucket name

# Initialize InfluxDB client
client = InfluxDBClient(url=url, token=token, org=org)
write_api = client.write_api()
query_api = client.query_api()


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


# which plants are going to be covered by clouds next
cloud_cover_next = []

stations = []
stations.append(station_provider(plant_to_node["plant1"], plant_to_node["plant2"]))
stations.append(station_provider(plant_to_node["plant2"], plant_to_node["plant3"]))

print("stations : ", stations)

def station_stream(station_list):
    print(f"Starting streaming for station list")
    print("lastnode:", station_list[-1]['id'])
    while(True):
        try:
            
            for station in station_list:

                station_index = station_list.index(station)
                if(station_index == 0):
                    # Query the last station
                    query = f'''
                    from(bucket: "{bucket}")
                    |> range(start: -30d)  // Adjust the range as needed
                    |> filter(fn: (r) => r._measurement == "sensor_grid_data")
                    |> filter(fn: (r) => int(v: r.id) == {station_list[-1]['id']})
                    |> limit(n: 1)
                    '''
                    
                    query_api = client.query_api()
                    # Execute the query
                    result = query_api.query(org=org, query=query)

                    # Extract the points
                    points = []
                    for table in result:
                        for record in table.records:
                            print("record: ", record)
                            points.append({
                                "id": record["id"],
                                "title": record["title"],
                                "lat": record["lat"],
                                "lon": record["lon"],
                                "color": record["color"],
                                "next": record["next"],
                                "irradiation": record["_value"],  # Use _value for irradiation
                                "time": record["_time"]
                            })
                    
                    points.sort(key=lambda x: x["time"], reverse=True)

                    # Get the last station point in line and recolor it if it was grey earlier
                    if points:
                        last_station_point = points[0]
                        if(last_station_point["color"] == "grey"):
                            recolored_point = (
                                Point("sensor_grid_data")  # Measurement name
                                .tag("id", int(last_station_point["id"]))  # Node Id
                                .tag("title", str(last_station_point["title"]))  # title for node
                                .tag("lat", float(last_station_point["lat"]))  # latitude of node
                                .tag("lon", float(last_station_point["lon"]))  # longitude of node
                                .tag("color", str("blue"))  # Color of node based on irradiation value
                                .tag("next", int(last_station_point["next"]))  # Color of node based on irradiation value
                                .field("irradiation", float(0.4))  # Irradiation
                                .time(datetime.utcnow())  # Current timestamp in UTC
                            )
                            write_api = client.write_api()
                            write_api.write(bucket=bucket, org=org, record=recolored_point)  # Write the point to InfluxDB
                    else:
                        print("No data points found for the given station.")


                if station["next"] in [1,2,3,4,5,6,7,8]:
                    # to which plant the cloud comes next
                    cloud_cover_next.append(station["next"])

                try:
                    
                    try:
                        # Query the point where "next" matches the current value
                        query = f"""
                        from(bucket: "{bucket}")
                        |> range(start: -30d)  // Adjust the range as needed
                        |> filter(fn: (r) => r._measurement == "sensor_grid_data")
                        |> filter(fn: (r) => int(v: r.next) == {station["id"]})
                        |> limit(n: 1)
                        """
                        
                        query_api = client.query_api()
                        result = query_api.query(org=org, query=query)
                        
                        points = []
                        for table in result:
                            for record in table.records:
                                print("record:", record)
                                points.append({
                                    "id": record["id"],
                                    "title": record["title"],
                                    "lat": record["lat"],
                                    "lon": record["lon"],
                                    "color": record["color"],
                                    "next": record["next"],
                                    "irradiation": record["_value"],  # Use _value for irradiation
                                    "time": record["_time"]
                                })
                        
                        points.sort(key=lambda x: x["time"], reverse=True)
                        
                        if not points:
                            print(f"No cloud coming to station with id: {station['id']}")
                            # Generate a random zero with a 50% chance
                            random_zero = random.choice([0, 1])  # Generates either 0 or 1
                            if(random_zero == 0):
                                irradiation = 0
                                current_station_color = "grey"
                            else:
                                irradiation = 0.4
                                current_station_color = "blue"

                            point = (
                                Point("sensor_grid_data")  # Measurement name
                                .tag("id", int(station["id"]))  # Node Id
                                .tag("title", str(station["title"]))  # title for node
                                .tag("lat", float(station["lat"]))  # latitude of node
                                .tag("lon", float(station["lon"]))  # longitude of node
                                .tag("color", str(current_station_color))  # Color of node based on irradiation value
                                .tag("next", int(station["next"]))  # Color of node based on irradiation value
                                .field("irradiation", float(irradiation))  # Irradiation
                                .time(datetime.utcnow())  # Current timestamp in UTC
                            )
                            write_api = client.write_api()
                            write_api.write(bucket=bucket, org=org, record=point)  # Write the point to InfluxDB
                            print(f"Data written for weather station: {point}")

                        else:
                            prev_point = points[0]

                            if prev_point["color"] == "grey":
                                # Recolor the previous point to "blue" if it was grey earlier
                                recolored_point = (
                                    Point("sensor_grid_data")  # Measurement name
                                    .tag("id", int(prev_point["id"]))  # Node Id
                                    .tag("title", str(prev_point["title"]))  # Title for node
                                    .tag("lat", float(prev_point["lat"]))  # Latitude of node
                                    .tag("lon", float(prev_point["lon"]))  # Longitude of node
                                    .tag("color", "blue")  # Color of node based on irradiation value
                                    .tag("next", int(prev_point["next"]))  # Next station ID
                                    .field("irradiation", 0.4)  # Irradiation
                                    .time(datetime.utcnow())  # Current timestamp in UTC
                                )

                                # Write the recolored point to InfluxDB
                                write_api = client.write_api()
                                write_api.write(bucket=bucket, org=org, record=recolored_point)
                                print(f"Recolored point written for station with id: {station['id']}")

                                irradiation = 0
                                current_station_color = "grey"

                            else:
                                print(f"Station with id {station['id']} already has color {prev_point['color']}, no update needed.")
                                # Generate a random zero with a 50% chance
                                random_zero = random.choice([0, 1])  # Generates either 0 or 1
                                if(random_zero == 0):
                                    irradiation = 0
                                    current_station_color = "grey"
                                else:
                                    irradiation = 0.4
                                    current_station_color = "blue"

                            
                            point = (
                                Point("sensor_grid_data")  # Measurement name
                                .tag("id", int(station["id"]))  # Node Id
                                .tag("title", str(station["title"]))  # title for node
                                .tag("lat", float(station["lat"]))  # latitude of node
                                .tag("lon", float(station["lon"]))  # longitude of node
                                .tag("color", str(current_station_color))  # Color of node based on irradiation value
                                .tag("next", int(station["next"]))  # Color of node based on irradiation value
                                .field("irradiation", float(irradiation))  # Irradiation
                                .time(datetime.utcnow())  # Current timestamp in UTC
                            )
                            write_api = client.write_api()
                            write_api.write(bucket=bucket, org=org, record=point)  # Write the point to InfluxDB
                            print(f"Data written for weather station: {point}")
                    
                    except Exception as e:
                        print(f"Error processing station {station['id']}: {e}")

                    time.sleep(60)  # Wait for 60 seconds before processing the next row
                except Exception as e:
                    print(f"Error while writing data: {e}")
                    traceback.print_exc()  # Print the traceback for debugging
                    continue
        except Exception as e:
            print(f"Error in station_stream: {e}")
            traceback.print_exc()  # Print the traceback for debugging
        
    print(f"Stopping streaming for station_stream") 


# Function to parse data from CSV and prepare for InfluxDB
def parse_csv_row_to_point(row):
    plant_node_id = int(plant_to_node[row["PLANT_ID"]]["id"])
    if plant_node_id in cloud_cover_next:
        row["IRRADIATION"] = 0
        cloud_cover_next.remove(plant_node_id)
 
    if(row["IRRADIATION"] > 0):
        color = "green"
    else:
        color = "red"
    return (
        Point("solar_generation_data")  # Measurement name
        .tag("id", plant_node_id)  # Node Id
        .tag("title", str(row["PLANT_ID"]))  # title for node
        .tag("lat", float(plant_to_node[row["PLANT_ID"]]["lat"]))  # latitude of node
        .tag("lon", float(plant_to_node[row["PLANT_ID"]]["lon"]))  # longitude of 
        .tag("color", str(color))  # Color of node based on irradiation value
        .tag("plant_id", str(row["PLANT_ID"]))  # Plant ID from CSV
        .field("ambient_temperature", float(row["AMBIENT_TEMPERATURE"]))  # Ambient temperature
        .field("module_temperature", float(row["MODULE_TEMPERATURE"]))  # Module temperature
        .field("irradiation", float(row["IRRADIATION"]))  # Irradiation
        .field("period_generation", float(row["PERIOD_GENERATION"]))  # Daily yield
        .time(datetime.utcnow())  # Current timestamp in UTC
    )


# Function to generate edge_server data for influxDB
def generate_edge_server_data():
    return (
        Point("solar_generation_data")  # Measurement name
        .tag("id", int(plant_to_node["edge_server1"]["id"]))  # Node Id
        .tag("title", "edge_server1")  # title for node
        .tag("lat", float(plant_to_node["edge_server1"]["lat"]))  # latitude of node
        .tag("lon", float(plant_to_node["edge_server1"]["lon"]))  # longitude of node
        .tag("color", "green")  # Color of node based on irradiation value
        .tag("plant_id", "edge_server1")  # Plant ID from CSV
        .field("ambient_temperature", float(28.0))  # Ambient temperature
        .field("module_temperature", float(28.0))  # Module temperature
        .field("irradiation", float(0.6))  # Irradiation
        .field("period_generation", float(20))  # Daily yield
        .time(datetime.utcnow())  # Current timestamp in UTC
    )

# Load CSV data into memory and filter by time range
def load_and_group_csv_data(file_path):
    with open(file_path, mode="r") as file:
        reader = csv.DictReader(file)
        grouped_data = defaultdict(list)
        for row in reader:
            try:
                # Parse and convert row fields to the required types
                row_time = datetime.strptime(row["DATE_TIME"], "%Y-%m-%d %H:%M:%S").time()
                plant_id = str(row["PLANT_ID"])
                ambient_temperature = float(row["AMBIENT_TEMPERATURE"])
                module_temperature = float(row["MODULE_TEMPERATURE"])
                irradiation = float(row["IRRADIATION"])
                period_generation = float(row["PERIOD_GENERATION"])

                # Filter rows by time range
                if dt_time(6, 0) <= row_time <= dt_time(18, 45):
                    grouped_data[plant_id].append({
                        "DATE_TIME": row["DATE_TIME"],
                        "PLANT_ID": plant_id,
                        "AMBIENT_TEMPERATURE": ambient_temperature,
                        "MODULE_TEMPERATURE": module_temperature,
                        "IRRADIATION": irradiation,
                        "PERIOD_GENERATION": period_generation,
                    })
            except (ValueError, KeyError) as e:
                print(f"Error processing row: {row} - {e}")
    return grouped_data


# Function to stream data for a single plant with error handling and loop termination
def plant_stream(plant_data):
    plant_id, rows = plant_data
    print(f"Starting streaming for {plant_id}")
    try:
        for row in rows:
            try:
                point = parse_csv_row_to_point(row)
                write_api = client.write_api()
                write_api.write(bucket=bucket, org=org, record=point)  # Write the point to InfluxDB
                print(f"Data written for {plant_id}: {point}")
                time.sleep(60)  # Wait for 60 seconds before processing the next row
            except Exception as e:
                print(f"Error while writing data for {plant_id}: {e}")
                traceback.print_exc()  # Print the traceback for debugging
                return  # Exit the loop and stop the thread
    except Exception as e:
        print(f"Error in plant_stream for {plant_id}: {e}")
        traceback.print_exc()  # Print the traceback for debugging
    finally:
        print(f"Stopping streaming for {plant_id}")

# Function to stream data for a single plant with error handling and loop termination
def edge_server_stream():
    print(f"Starting streaming for edge_server")
    try:
        try:
            point = generate_edge_server_data()
            write_api = client.write_api()
            write_api.write(bucket=bucket, org=org, record=point)  # Write the point to InfluxDB
            print(f"Data written for edge_server: {point}")
            time.sleep(60)  # Wait for 60 seconds before processing the next row
        except Exception as e:
            print(f"Error while writing data for edge_server: {e}")
            traceback.print_exc()  # Print the traceback for debugging
            return  # Exit the loop and stop the thread
    except Exception as e:
        print(f"Error in plant_stream for edge_server: {e}")
        traceback.print_exc()  # Print the traceback for debugging
    finally:
        print(f"Stopping streaming for edge_server")

# Wrapper to catch and log exceptions in threads
def thread_wrapper(func, *args, **kwargs):
    try:
        func(*args, **kwargs)
    except Exception as e:
        print(f"Unhandled exception in thread: {e}")
        traceback.print_exc()  # Print the traceback for debugging

# Main streaming function
def stream_data_from_csv(file_path):
    grouped_data = load_and_group_csv_data(file_path)
    try:
        print("Streaming data from CSV to InfluxDB... Press Ctrl+C to stop.")
        
        with ThreadPoolExecutor(max_workers=len(grouped_data) + 1 + len(stations)) as executor:
            # Store futures in a list
            futures = []

            # Stream plant data
            for plant_data in grouped_data.items():
                futures.append(executor.submit(thread_wrapper, plant_stream, plant_data))

            # Run edge_server_stream in a separate thread
            futures.append(executor.submit(thread_wrapper, edge_server_stream))

            # Stream station data
            for station_list in stations:
                futures.append(executor.submit(thread_wrapper, station_stream, station_list))

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


# Path to the CSV file
csv_file_path = "data/part_1.csv"

# Start streaming data
stream_data_from_csv(csv_file_path)
