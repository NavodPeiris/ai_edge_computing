import pandas as pd
from influxdb import InfluxDBClient

# InfluxDB connection parameters
host = "localhost"  # InfluxDB host
port = 8086         # InfluxDB port
username = "navod"  # Username for InfluxDB
password = "1234"  # Password for InfluxDB
database = "solar_power_generation"  # Database name

# Initialize InfluxDB client
client = InfluxDBClient(host=host, port=port, username=username, password=password, database=database)

client.drop_measurement("solar_generation_data")
client.drop_measurement("solar_generation_pred")

# Close the client
client.close()

print("Data delete complete.")
