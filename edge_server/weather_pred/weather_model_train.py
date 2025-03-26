import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
import joblib

def train_multi_output_model():
    data_path = 'weather_data/testset.csv'

    # Load and preprocess the data
    df = pd.read_csv(data_path)
    df.columns = df.columns.str.strip()  # Remove spaces in column names
    df.index = pd.to_datetime(df['datetime_utc'])  # Set datetime as index
    required_cols = ['humidity', 'rain', 'temperature']
    df = df[required_cols]
    df = df.fillna(method='ffill')  # Fill missing values
    df_final = df.resample('D').mean().fillna(method='ffill')  # Resample daily and fill again

    # Normalize the data
    scaler = MinMaxScaler()
    scaled_data = scaler.fit_transform(df_final)
    # Save the scaler
    joblib.dump(scaler, 'weather_pred_models/scaler.pkl')

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
    model.add(Dense(units=3))  # Output layer for 3 targets (humidity, rain, temperature)

    # Compile the model
    model.compile(optimizer='adam', loss='mean_squared_error')

    # Define callbacks
    early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
    model_checkpoint = ModelCheckpoint('weather_pred_models/hum_rain_temp.h5', monitor='val_loss', save_best_only=True)

    # Train the model
    history = model.fit(
        train_x, train_y,
        epochs=100,
        batch_size=64,
        validation_split=0.2,
        callbacks=[early_stopping, model_checkpoint]
    )

    # Evaluate the model on the test set
    best_model = tf.keras.models.load_model('weather_pred_models/hum_rain_temp.h5')
    test_loss = best_model.evaluate(test_x, test_y)
    print("Test Loss:", test_loss)

    # Make predictions
    predictions = best_model.predict(test_x)

    # Inverse transform to get original scale
    true_values = scaler.inverse_transform(test_y)[:, :3]
    predicted_values = scaler.inverse_transform(predictions)[:, :3]

    # Plot predictions vs actuals for the last 100 data points
    plt.figure(figsize=(12, 8))
    plt.plot(true_values[-100:, 0], label='Actual Humidity')
    plt.plot(predicted_values[-100:, 0], label='Predicted Humidity')
    plt.plot(true_values[-100:, 1], label='Actual Rain')
    plt.plot(predicted_values[-100:, 1], label='Predicted Rain')
    plt.plot(true_values[-100:, 2], label='Actual Temperature')
    plt.plot(predicted_values[-100:, 2], label='Predicted Temperature')
    plt.legend()
    plt.title("Predicted vs Actual for Humidity, Rain, and Temperature")
    plt.xlabel('Time (days)')
    plt.ylabel('Value')
    plt.show()


def infer_multi_output(data):
    # Normalize the data
    # Load the saved scaler
    scaler = joblib.load('weather_pred_models/scaler.pkl')
    scaled_data = scaler.transform(data)
    reshaped_data = np.expand_dims(scaled_data, axis=0)  # Add a batch dimension

    # Load the trained model
    best_model = tf.keras.models.load_model('weather_pred_models/hum_rain_temp.h5')
    predictions = best_model.predict(reshaped_data)
    
    # Inverse transform the predictions
    predicted_values = scaler.inverse_transform(predictions)[-1]

    # Format the predicted values to two decimal points
    formatted_values = [round(value, 2) for value in predicted_values]

    return formatted_values


# Train the model
train_multi_output_model()
