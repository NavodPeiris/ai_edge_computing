from sklearn.model_selection import train_test_split
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
bucket = "solar_power_generation"  # Bucket name

# Initialize InfluxDB client
client = InfluxDBClient(url=url, token=token, org=org)

query_api = client.query_api()
write_api = client.write_api()

plant_ids = [
    "plant1", "plant2", "plant3"
]

# Function to fetch points from InfluxDB
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
        
        return latest_data

    except Exception as e:
        print(f"Error fetching data: {e}")
        return pd.DataFrame(), None
    

def train_model(data, plant_id):
    
    # Normalize the data (we'll normalize only the features)
    scaler = MinMaxScaler()
    scaled_data = scaler.fit_transform(data[['AMBIENT_TEMPERATURE', 'MODULE_TEMPERATURE', 'IRRADIATION', 'PERIOD_GENERATION']])

    os.makedirs(f"/opt/airflow/dags/power_pred_models/{plant_id}", exist_ok=True)

    # Save the scaler
    joblib.dump(scaler, f'/opt/airflow/dags/power_pred_models/{plant_id}/scaler.pkl')

    # Replace the original data with the scaled data
    data[['AMBIENT_TEMPERATURE', 'MODULE_TEMPERATURE', 'IRRADIATION', 'PERIOD_GENERATION']] = scaled_data

    # Create the features and target columns
    def create_sequences(df, sequence_length=4):
        features = []
        targets = []
        
        # Loop through each row, starting from the `sequence_length` index
        for i in range(sequence_length, len(df)):
            # Get the previous `sequence_length` rows as features
            feature_data = df.iloc[i-sequence_length:i][['AMBIENT_TEMPERATURE', 'MODULE_TEMPERATURE', 'IRRADIATION']]
            features.append(feature_data.values)
            
            # Get the target value (DAILY_YIELD of the next row)
            target = df.iloc[i]['PERIOD_GENERATION']
            targets.append(target)
        
        return np.array(features), np.array(targets)

    # Create sequences
    X, y = create_sequences(data)

    # Reshape X to be 3D for LSTM input (samples, timesteps, features)
    X = X.reshape((X.shape[0], X.shape[1], X.shape[2]))

    print(f"Input shape: {X.shape}")
    print(f"Output shape: {y.shape}")

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    print(f"Training set shape: {X_train.shape}")
    print(f"Testing set shape: {X_test.shape}")

    # Build the LSTM model
    model = Sequential()

    # Add LSTM layer
    model.add(LSTM(units=64, return_sequences=False, input_shape=(X.shape[1], X.shape[2])))

    # Add Dropout for regularization
    model.add(Dropout(0.2))

    # Add Dense layer for output
    model.add(Dense(1))  # Output layer with 1 unit (DAILY_YIELD)

    # Compile the model
    model.compile(optimizer='adam', loss='mean_squared_error')

    # Define callbacks
    early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
    model_checkpoint = ModelCheckpoint(f'/opt/airflow/dags/power_pred_models/{plant_id}/model.h5', monitor='val_loss', save_best_only=True)

    # Train the model
    history = model.fit(
        X_train, y_train,
        epochs=100,
        batch_size=64,
        validation_split=0.2,
        callbacks=[early_stopping, model_checkpoint]
    )

    # Evaluate the model on the test set
    best_model = tf.keras.models.load_model(f'/opt/airflow/dags/power_pred_models/{plant_id}/model.h5')
    test_loss = best_model.evaluate(X_test, y_test)
    print("Test Loss:", test_loss)


default_args = {
    'owner': 'admin',
    'retries': 5,
    'retry_delay': timedelta(minutes=2)
}

def train(ti):
    for plant_id in plant_ids:
        data = fetch_latest_data(plant_id)
        train_model(data, plant_id)


with DAG(
    default_args=default_args,
    dag_id='train_power_pred',
    description='Train Solar Power Generation Prediction Model',
    start_date=datetime(2025, 7, 21, 0),
    schedule_interval='@daily'
) as dag:
    
    task1 = PythonOperator(
        task_id='train',
        python_callable=train,
    )

    task1
    