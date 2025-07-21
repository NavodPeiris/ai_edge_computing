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
from sklearn.preprocessing import OneHotEncoder
from sklearn.cluster import KMeans
from keras.losses import MeanSquaredError

def load_and_preprocess_data(df, model_path, deliver_scaler_url, deliver_encoder_url):
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

    base_path = "/".join(model_path.split("/")[:-1])

    encoder_path = f"{base_path}/encoder.pkl"

    # fetch encoder if not available locally
    if not os.path.exists(encoder_path):
        response = requests.post(deliver_encoder_url, json={"encoder_path": encoder_path})

        if response.status_code == 200:
            with open(encoder_path, "wb") as f:
                f.write(response.content)
            print(f"file saved as {encoder_path}")
        else:
            print(f"Error: {response.json()['detail']}")

    with open(encoder_path, "rb") as f:
        encoder = pickle.load(f)

    categorical_cols = df.select_dtypes(include=['object', 'category']).columns
   
    # 1. Get encoded feature names from OneHotEncoder
    encoded_cat_cols = encoder.get_feature_names_out(categorical_cols)

    # 2. Get numeric column names
    num_cols = df.drop(columns=categorical_cols).columns

    df[categorical_cols] = df[categorical_cols].astype(str) # make all values in categorical columns str as encoder need uniform types

    # 3. Combine data
    X_cat = encoder.transform(df[categorical_cols])
    X_num = df[num_cols].values  # keep column order
    X_final = np.hstack((X_num, X_cat))

    # 4. Convert to DataFrame with correct column names
    X_final_df = pd.DataFrame(X_final, columns=list(num_cols) + list(encoded_cat_cols))


    # Scale only original numerical features in X (exclude datetime columns)
    num_cols = df.select_dtypes(include=['number']).columns.difference(datetime_cols)

    scaler_path = f"{base_path}/scaler.pkl"

    # fetch scaler if not available locally
    if not os.path.exists(scaler_path):
        response = requests.post(deliver_scaler_url, json={"scaler_path": scaler_path})

        if response.status_code == 200:
            with open(scaler_path, "wb") as f:
                f.write(response.content)
            print(f"file saved as {scaler_path}")
        else:
            print(f"Error: {response.json()['detail']}")
    
    with open(scaler_path, "rb") as f:
        scaler = pickle.load(f)

    X_final_df[num_cols] = scaler.fit_transform(X_final_df[num_cols])

    # Convert to numpy arrays
    X_res = np.array(X_final_df, dtype=np.float32)

    return X_res



def infer(df, labels, task_type, model_path, server_url):
    base_path = "/".join(model_path.split("/")[:-1])
    os.makedirs(base_path, exist_ok=True)  
    
    # Convert DataFrame to NumPy array and ensure correct dtype
    X = load_and_preprocess_data(df, model_path, f"{server_url}/deliver_scaler/", f"{server_url}/deliver_encoder/")

    # if model is not available, deliver the model
    if not os.path.exists(model_path):
        response = requests.post(f"{server_url}/deliver_model/", json={"model_path": model_path})

        if response.status_code == 200:
            with open(model_path, "wb") as f:
                f.write(response.content)
            print(f"Model saved as {model_path}")
        else:
            print(f"Error: {response.json()['detail']}")

    if task_type == "classification":
        # Load the model
        model = load_model(model_path)

        # Perform inference
        predictions = model.predict(X).flatten()  # Flatten in case it's a 2D array

        # Apply thresholding
        predictions = np.where(predictions > 0.5, 1, 0)

        if len(labels) > 1:
            for i, label in enumerate(labels):
                # Append predictions as a new column to the DataFrame
                df[label] = predictions[:, i] if predictions.ndim > 1 else predictions
        else:
            # Append predictions as a new column to the DataFrame
            df[labels[0]] = predictions 

    elif task_type == "regression" or task_type == "forecasting":
        # Load the model
        model = load_model(model_path, custom_objects={ "mse": MeanSquaredError() })

        # Perform inference
        predictions = model.predict(X)

        if len(labels) > 1:
            for i, label in enumerate(labels):
                # Append predictions as a new column to the DataFrame
                df[label] = predictions[:, i] if predictions.ndim > 1 else predictions
        else:
            # Append predictions as a new column to the DataFrame
            df[labels[0]] = predictions.flatten()  # Flatten in case it's a 2D array 

    elif task_type == "unsupervised classification":
        # Load the trained autoencoder model
        autoencoder = load_model(model_path, custom_objects={ "mse": MeanSquaredError() })

        # Extract encoder (all layers up to latent representation)
        encoder = Model(inputs=autoencoder.input, outputs=autoencoder.layers[2].output)

        latent_features = encoder.predict(X)

        kmeans = KMeans(n_clusters=len(labels), random_state=42)
        class_ids = kmeans.fit_predict(latent_features)

        class_names = []

        for class_id in class_ids:
            class_names.append(labels[class_id])

        df["class"] = class_names

    elif task_type == "anomaly detection":
        # Load the trained autoencoder model
        autoencoder = load_model(model_path, custom_objects={ "mse": MeanSquaredError() })

        # Predict reconstruction (output) for input data X
        X_reconstructed = autoencoder.predict(X)

        # Compute the Mean Squared Error (MSE) for each sample
        reconstruction_error = np.mean(np.square(X - X_reconstructed), axis=1)

        # Set anomaly detection threshold (e.g., 95th percentile of reconstruction error)
        threshold = np.percentile(reconstruction_error, 95)

        # Label samples: 1 = anomaly, 0 = normal
        anomaly_labels = (reconstruction_error > threshold).astype(int)

        df[labels[0]] = anomaly_labels

    return df
