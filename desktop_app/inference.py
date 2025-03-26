import requests
import tensorflow as tf
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
import pickle
import os
from tensorflow.keras.models import load_model
from tensorflow.keras import Model
from tensorflow.keras.layers import Input
import umap


def load_and_preprocess_data(df, model_path, deliver_scaler_url):
    for col in df.columns:
        if df[col].isnull().sum() > 0:
            if df[col].dtype == 'object':  # Categorical feature2
                df[col].fillna(df[col].mode()[0], inplace=True)
            else:  # Numerical feature
                df[col].fillna(df[col].median(), inplace=True)

    # Identify datetime columns correctly
    datetime_cols = [col for col in df.columns if np.issubdtype(df[col].dtype, np.datetime64)]

    # Convert datetime columns to numerical values (Unix timestamp)
    for col in datetime_cols:
        df[col] = df[col].astype(np.int64) // 10**9

    # One-hot encode categorical features in X
    X_encoded = pd.get_dummies(df, drop_first=True)

    # Scale only original numerical features in X (exclude datetime columns)
    num_cols = df.select_dtypes(include=['number']).columns.difference(datetime_cols)
    base_path = "/".join(model_path.split("/")[:-1])

    scaler_path = f"{base_path}/scaler.pkl"

    if not os.path.exists(scaler_path):
        response = requests.post(deliver_scaler_url, json={"scaler_path": scaler_path})

        if response.status_code == 200:
            with open(scaler_path, "wb") as f:
                f.write(response.content)
            print(f"Model saved as {scaler_path}")
        else:
            print(f"Error: {response.json()['detail']}")
    
    with open(scaler_path, "rb") as f:
        scaler = pickle.load(f)

    X_encoded[num_cols] = scaler.fit_transform(X_encoded[num_cols])
    X_encoded.to_csv("infer.csv")

    # Convert to numpy arrays
    X_res = np.array(X_encoded, dtype=np.float32)

    return X_res



def infer(df, label, task_type, model_path, server_url):
    base_path = "/".join(model_path.split("/")[:-1])
    os.makedirs(base_path, exist_ok=True)  
    
    # Convert DataFrame to NumPy array and ensure correct dtype
    X = load_and_preprocess_data(df, model_path, f"{server_url}/deliver_scaler/")

    if not os.path.exists(model_path):
        response = requests.post(f"{server_url}/deliver_model/", json={"model_path": model_path})

        if response.status_code == 200:
            with open(model_path, "wb") as f:
                f.write(response.content)
            print(f"Model saved as {model_path}")
        else:
            print(f"Error: {response.json()['detail']}")

    if task_type == "classification" or task_type == "regression":
        # Load the model
        model = load_model(model_path)

        # Perform inference
        predictions = model.predict(X).flatten()  # Flatten in case it's a 2D array

        # Apply thresholding
        predictions = np.where(predictions > 0.5, 1, 0)

        # Append predictions as a new column to the DataFrame
        df[label] = predictions
    elif task_type == "unsupervised classification":
        # Load the trained autoencoder model
        autoencoder = load_model(model_path)

        # Extract encoder (all layers up to latent representation)
        encoder = Model(inputs=autoencoder.input, outputs=autoencoder.layers[2].output)

        latent_features = encoder.predict(X)

        # Apply UMAP to reduce dimensionality to 2D
        umap_model = umap.UMAP(n_components=2, random_state=42)
        latent_2d = umap_model.fit_transform(latent_features)

        # Convert X (original data) to a DataFrame for easy visualization
        df = pd.DataFrame(X, columns=[f"Feature_{i}" for i in range(X.shape[1])])

        # Add UMAP coordinates to DataFrame
        df["umap_1"] = latent_2d[:, 0]
        df["umap_2"] = latent_2d[:, 1]

        # Create a hover text column showing the actual data record
        df["hover_text"] = df.apply(lambda row: "<br>".join([f"{col}: {row[col]:.2f}" for col in df.columns[:-2]]), axis=1)
    elif task_type == "anomaly detection":
        # Load the trained autoencoder model
        autoencoder = load_model(model_path)

        # Predict reconstruction (output) for input data X
        X_reconstructed = autoencoder.predict(X)

        # Compute the Mean Squared Error (MSE) for each sample
        reconstruction_error = np.mean(np.square(X - X_reconstructed), axis=1)

        # Set anomaly detection threshold (e.g., 95th percentile of reconstruction error)
        threshold = np.percentile(reconstruction_error, 95)

        # Label samples: 1 = anomaly, 0 = normal
        anomaly_labels = (reconstruction_error > threshold).astype(int)

        df[label] = anomaly_labels

    return df
