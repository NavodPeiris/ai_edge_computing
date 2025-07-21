from airflow import DAG

from datetime import datetime, timedelta
from airflow.operators.python import PythonOperator

import os
from influxdb_client import InfluxDBClient
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import joblib
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
import matplotlib.pyplot as plt

# InfluxDB connection parameters
url = "http://influxdb_fyp:8086"  # InfluxDB 2.x URL
token = "3wvWUxmtdBM03hm9YgTEa91s6ofQ73G4gQ54uNR0Ek59zpJNMGOagj1UR1GKw3D1f5Elw-zS78rEwY7akZGmOw=="  # Authentication token
org = "fyp"      # Organization name
bucket = "weather_data"  # Bucket name

# Initialize InfluxDB client
client = InfluxDBClient(url=url, token=token, org=org)

query_api = client.query_api()
write_api = client.write_api()

#locations = ["malabe", "kandy", "mount lavinia", "maharagama"]
locations = ["malabe"]

# Function to fetch points from InfluxDB
def fetch_latest_data(location):
    
    query = f'''
    from(bucket: "{bucket}")
        |> range(start: -30d)  // Adjust the range as needed
        |> filter(fn: (r) => r._measurement == "weather_data_gen")
        |> filter(fn: (r) => r.location == "{location}")  // Filter by location tag
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
                    "humidity": record["humidity"],
                    "rain": record["rain"],
                    "temperature": record["temperature"],
                    "time": record["_time"].date()
                })

        points.sort(key=lambda x: x["time"], reverse=True)

        # If no points found, return empty DataFrame and None
        if not points:
            return pd.DataFrame(), None
        

        processed_points = []
        for point in points:
            processed_points.append({
                "humidity": point["humidity"],
                "rain": point["rain"],
                "temperature": point["temperature"]
            })
        
        # Convert the points to a pandas DataFrame
        latest_data = pd.DataFrame(processed_points)
        
        return latest_data

    except Exception as e:
        print(f"Error fetching data: {e}")
        return pd.DataFrame(), None
    

def train_model(data, location):
    
    # Normalize the data (we'll normalize only the features)
    scaler = MinMaxScaler()
    scaled_data = scaler.fit_transform(data[['humidity', 'rain', 'temperature']])

    os.makedirs(f"/opt/airflow/dags/weather_pred_models/{location}", exist_ok=True)

    # Save the scaler
    joblib.dump(scaler, f'/opt/airflow/dags/weather_pred_models/{location}/scaler.pkl')

    # Replace the original data with the scaled data
    data[['humidity', 'rain', 'temperature']] = scaled_data

    # Create sequences and labels
    sequence_length = 7  # Number of time steps in each sequence
    sequences = []
    labels = []

    for i in range(len(scaled_data) - sequence_length):
        seq = scaled_data[i:i + sequence_length]
        label = scaled_data[i + sequence_length]  # Multi-output: all three values
        sequences.append(seq)
        labels.append(label)

    # Convert to numpy arrays
    sequences = np.array(sequences)
    labels = np.array(labels)

    # Split into train and test sets
    train_size = int(0.8 * len(sequences))
    train_x, test_x = sequences[:train_size], sequences[train_size:]
    train_y, test_y = labels[:train_size], labels[train_size:]

    print("Train X shape:", train_x.shape)
    print("Train Y shape:", train_y.shape)
    print("Test X shape:", test_x.shape)
    print("Test Y shape:", test_y.shape)

    # Build the model
    model = Sequential()
    model.add(LSTM(units=128, input_shape=(train_x.shape[1], train_x.shape[2]), return_sequences=True))
    model.add(Dropout(0.2))
    model.add(LSTM(units=64, return_sequences=False))
    model.add(Dropout(0.2))
    model.add(Dense(units=3))  # Output layer for predicted humidity, rain, temperature

    # Compile the model
    model.compile(optimizer='adam', loss='mean_squared_error')

    # Define callbacks
    early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
    model_checkpoint = ModelCheckpoint(f'/opt/airflow/dags/weather_pred_models/{location}/model.h5', monitor='val_loss', save_best_only=True)

    # Train the model
    history = model.fit(
        train_x, train_y,
        epochs=100,
        batch_size=64,
        validation_split=0.2,
        callbacks=[early_stopping, model_checkpoint]
    )

    # Evaluate the model on the test set
    best_model = tf.keras.models.load_model(f'/opt/airflow/dags/weather_pred_models/{location}/model.h5')
    test_loss = best_model.evaluate(test_x, test_y)
    print("Test Loss:", test_loss)


default_args = {
    'owner': 'admin',
    'retries': 5,
    'retry_delay': timedelta(minutes=2)
}

def train(ti):
    for location in locations:
        data = fetch_latest_data(location)
        train_model(data, location)


with DAG(
    default_args=default_args,
    dag_id='train_weather_pred',
    description='Train Weather Prediction Model',
    start_date=datetime(2025, 7, 21, 0),
    schedule_interval='@daily'
) as dag:
    
    task1 = PythonOperator(
        task_id='train',
        python_callable=train,
    )

    task1
    