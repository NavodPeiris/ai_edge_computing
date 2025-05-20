from influxdb_client import InfluxDBClient
from influxdb_client.client.write_api import SYNCHRONOUS

# Replace these variables with your InfluxDB configuration
url = "http://localhost:8086"  # InfluxDB URL
token = "3wvWUxmtdBM03hm9YgTEa91s6ofQ73G4gQ54uNR0Ek59zpJNMGOagj1UR1GKw3D1f5Elw-zS78rEwY7akZGmOw=="  # Authentication token
org = "fyp"      # Organization name

# Initialize the InfluxDB client
client = InfluxDBClient(url=url, token=token, org=org)

# Execute the delete operation
try:
    # Delete data using the delete API
    delete_api = client.delete_api()
    delete_api.delete(
        start="1900-01-01T00:00:00Z",
        stop="2100-01-01T00:00:00Z",
        predicate=f'_measurement="solar_generation_data"',
        bucket="solar_power_generation",
        org=org,
    )
    

    # Delete data using the delete API
    delete_api = client.delete_api()
    delete_api.delete(
        start="1900-01-01T00:00:00Z",
        stop="2100-01-01T00:00:00Z",
        predicate=f'_measurement="solar_generation_pred"',
        bucket="solar_power_generation",
        org=org,
    )
    

    # Delete data using the delete API
    delete_api = client.delete_api()
    delete_api.delete(
        start="1900-01-01T00:00:00Z",
        stop="2100-01-01T00:00:00Z",
        predicate=f'_measurement="sensor_grid_data"',
        bucket="solar_power_generation",
        org=org,
    )


    # Delete data using the delete API
    delete_api = client.delete_api()
    delete_api.delete(
        start="1900-01-01T00:00:00Z",
        stop="2100-01-01T00:00:00Z",
        predicate=f'_measurement="weather_data_gen"',
        bucket="weather_data",
        org=org,
    )


    # Delete data using the delete API
    delete_api = client.delete_api()
    delete_api.delete(
        start="1900-01-01T00:00:00Z",
        stop="2100-01-01T00:00:00Z",
        predicate=f'_measurement="weather_data_pred"',
        bucket="weather_data",
        org=org,
    )


    # Delete data using the delete API
    delete_api = client.delete_api()
    delete_api.delete(
        start="1900-01-01T00:00:00Z",
        stop="2100-01-01T00:00:00Z",
        predicate=f'_measurement="people_count"',
        bucket="common_camera_data",
        org=org,
    )


    # Delete data using the delete API
    delete_api = client.delete_api()
    delete_api.delete(
        start="1900-01-01T00:00:00Z",
        stop="2100-01-01T00:00:00Z",
        predicate=f'_measurement="vehicles_count"',
        bucket="traffic_camera_data",
        org=org,
    )


    # Delete data using the delete API
    delete_api = client.delete_api()
    delete_api.delete(
        start="1900-01-01T00:00:00Z",
        stop="2100-01-01T00:00:00Z",
        predicate=f'_measurement="pred_people_count"',
        bucket="common_camera_data",
        org=org,
    )

    # Delete data using the delete API
    delete_api = client.delete_api()
    delete_api.delete(
        start="1900-01-01T00:00:00Z",
        stop="2100-01-01T00:00:00Z",
        predicate=f'_measurement="pred_vehicles_count"',
        bucket="traffic_camera_data",
        org=org,
    )
    

except Exception as e:
    print(f"Failed to delete measurement: {e}")
finally:
    client.close()
