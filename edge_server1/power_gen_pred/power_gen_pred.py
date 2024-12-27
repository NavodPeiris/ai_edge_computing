import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import joblib
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

def train_model():
    # Load the data
    data = pd.read_csv("data/part_1.csv")

    # Convert DATE_TIME to datetime
    data['DATE_TIME'] = pd.to_datetime(data['DATE_TIME'])

    # Sort data by DATE_TIME for each PLANT_ID
    data = data.sort_values(by=['PLANT_ID', 'DATE_TIME'])

    # Normalize the data (we'll normalize only the features)
    scaler = MinMaxScaler()
    scaled_data = scaler.fit_transform(data[['AMBIENT_TEMPERATURE', 'MODULE_TEMPERATURE', 'IRRADIATION', 'DAILY_YIELD']])

    # Save the scaler
    joblib.dump(scaler, 'power_pred_models/scaler.pkl')

    # Replace the original data with the scaled data
    data[['AMBIENT_TEMPERATURE', 'MODULE_TEMPERATURE', 'IRRADIATION', 'DAILY_YIELD']] = scaled_data

    # Create the features and target columns
    def create_sequences(df, sequence_length=4):
        features = []
        targets = []
        
        # Loop through each plant
        for plant_id in df['PLANT_ID'].unique():
            plant_data = df[df['PLANT_ID'] == plant_id].reset_index(drop=True)
            
            # Loop through each row, starting from the `sequence_length` index
            for i in range(sequence_length, len(plant_data)):
                # Get the previous `sequence_length` rows as features
                feature_data = plant_data.iloc[i-sequence_length:i][['AMBIENT_TEMPERATURE', 'MODULE_TEMPERATURE', 'IRRADIATION']]
                features.append(feature_data.values)
                
                # Get the target value (DAILY_YIELD of the next row)
                target = plant_data.iloc[i]['DAILY_YIELD']
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
    model.compile(optimizer=Adam(), loss='mean_squared_error')

    # Define callbacks
    early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
    model_checkpoint = ModelCheckpoint('power_pred_models/power_pred.h5', monitor='val_loss', save_best_only=True)

    # Train the model
    model.fit(X_train, y_train, epochs=10, batch_size=32, validation_split=0.2, callbacks=[early_stopping, model_checkpoint])

    # Evaluate the model
    loss = model.evaluate(X_test, y_test)
    print(f"Loss: {loss}")


def infer_model(data):
    # Normalize the data
    # Load the saved scaler
    scaler = joblib.load('power_pred_models/scaler.pkl')
    scaled_data = scaler.transform(data[['AMBIENT_TEMPERATURE', 'MODULE_TEMPERATURE', 'IRRADIATION', 'DAILY_YIELD']])
    scaled_data = np.delete(scaled_data, -1, axis=1)
    reshaped_data = np.expand_dims(scaled_data, axis=0)  # Add a batch dimension

    # Load the trained model
    best_model = tf.keras.models.load_model('power_pred_models/power_pred.h5')
    predictions = best_model.predict(reshaped_data)

    # Repeat the prediction value 4 times to create a 1x4 array
    predictions_reshaped = np.repeat(predictions, 4).reshape(1, 4)

    print(predictions_reshaped.shape)

    # Now inverse transform
    true_pred = scaler.inverse_transform(predictions_reshaped)[-1][-1]

    rounded_daily_yield = round(true_pred, 2)
    return rounded_daily_yield
