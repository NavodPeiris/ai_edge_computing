import tensorflow as tf
import numpy as np
import joblib
import pandas as pd

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


'''
# Example inference
data_path = 'weather_data/testset.csv'
df = pd.read_csv(data_path)
df.columns = df.columns.str.strip()
df.index = pd.to_datetime(df['datetime_utc'])
df = df[['humidity', 'rain', 'temperature']].fillna(method='ffill')
df_final = df.resample('D').mean().fillna(method='ffill')

# Predict using the first 7 days of data
predicted_values = infer_multi_output(df_final.head(7))


print("Predicted values (Humidity, Rain, Temperature):", predicted_values)
humidity_pred = predicted_values[0]
rain_pred = predicted_values[1] >= 0.7
temp_pred = predicted_values[2]'
'''