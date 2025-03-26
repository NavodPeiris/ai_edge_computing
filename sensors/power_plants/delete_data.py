from influxdb_client import InfluxDBClient
from influxdb_client.client.write_api import SYNCHRONOUS

# Replace these variables with your InfluxDB configuration
url = "http://localhost:8086"  # InfluxDB URL
token = "3wvWUxmtdBM03hm9YgTEa91s6ofQ73G4gQ54uNR0Ek59zpJNMGOagj1UR1GKw3D1f5Elw-zS78rEwY7akZGmOw=="  # Authentication token
org = "fyp"      # Organization name
bucket = "solar_power_generation"  # Bucket name
gen_measurement_name = "solar_generation_data"  # Measurement to delete
pred_measurement_name = "solar_generation_pred"  # Measurement to delete
sensor_grid = "sensor_grid_data"

# Initialize the InfluxDB client
client = InfluxDBClient(url=url, token=token, org=org)

# Execute the delete operation
try:
    # Delete data using the delete API
    delete_api = client.delete_api()
    delete_api.delete(
        start="1900-01-01T00:00:00Z",
        stop="2100-01-01T00:00:00Z",
        predicate=f'_measurement="{gen_measurement_name}"',
        bucket=bucket,
        org=org,
    )
    print(f"Measurement '{gen_measurement_name}' successfully deleted from bucket '{bucket}'.")

    # Delete data using the delete API
    delete_api = client.delete_api()
    delete_api.delete(
        start="1900-01-01T00:00:00Z",
        stop="2100-01-01T00:00:00Z",
        predicate=f'_measurement="{pred_measurement_name}"',
        bucket=bucket,
        org=org,
    )
    print(f"Measurement '{pred_measurement_name}' successfully deleted from bucket '{bucket}'.")

    # Delete data using the delete API
    delete_api = client.delete_api()
    delete_api.delete(
        start="1900-01-01T00:00:00Z",
        stop="2100-01-01T00:00:00Z",
        predicate=f'_measurement="{sensor_grid}"',
        bucket=bucket,
        org=org,
    )
    print(f"Measurement '{sensor_grid}' successfully deleted from bucket '{bucket}'.")

except Exception as e:
    print(f"Failed to delete measurement: {e}")
finally:
    client.close()
