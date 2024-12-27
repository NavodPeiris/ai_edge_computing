import pandas as pd
from influxdb import InfluxDBClient

# InfluxDB connection parameters
host = "localhost"  # InfluxDB host
port = 8086         # InfluxDB port
username = "navod"  # Username for InfluxDB
password = "1234"  # Password for InfluxDB
database = "power_data"  # Database name

# Initialize InfluxDB client
client = InfluxDBClient(host=host, port=port, username=username, password=password, database=database)

# Read the CSV file into a pandas DataFrame
csv_file = "data/part_1.csv"
df = pd.read_csv(csv_file)

# Iterate over the rows of the DataFrame and write data to InfluxDB
json_body = []

# Assuming your CSV has columns: DATE_TIME, AMBIENT_TEMPERATURE, MODULE_TEMPERATURE, IRRADIATION, PLANT_ID, DAILY_YIELD
for index, row in df.iterrows():
    # Create a JSON body for each row
    point = {
        "measurement": "solar_data",  # Measurement name
        "tags": {
            "plant_id": row['PLANT_ID'],  # Tag field
        },
        "fields": {
            "ambient_temperature": float(row['AMBIENT_TEMPERATURE']),
            "module_temperature": float(row['MODULE_TEMPERATURE']),
            "irradiation": float(row['IRRADIATION']),
            "daily_yield": float(row['DAILY_YIELD']),
        },
        "time": row['DATE_TIME']  # Timestamp
    }
    json_body.append(point)

# Write the data to InfluxDB
client.write_points(json_body)

# Close the client
client.close()

print("Data ingestion complete.")
