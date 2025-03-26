# client.py
import flwr as fl
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
from imblearn.over_sampling import SMOTE
from tensorflow.keras import layers, models
import requests
import time
import argparse
import pickle
import os

class FederatedClient(fl.client.NumPyClient):
    def __init__(self, df, save_path, task_type, label):
        self.df = df
        self.save_path = save_path
        self.task_type = task_type
        self.label = label

        self.model = None
        self.X_train, self.X_test, self.y_train, self.y_test, self.input_dim, self.output_dim = self.load_and_preprocess_data()
        print("input: ", self.input_dim)
        print("output: ", self.output_dim)

    def load_and_preprocess_data(self):
    
        # Separate features and target
        if self.label != "":
            X = self.df.drop(columns=[self.label])
        else:
            X = self.df

        if self.label != "":
            y = self.df[self.label]

        # Fill missing values in X
        for col in X.columns:
            if X[col].isnull().sum() > 0:
                if X[col].dtype == 'object':  # Categorical feature2
                    X[col].fillna(X[col].mode()[0], inplace=True)
                else:  # Numerical feature
                    X[col].fillna(X[col].median(), inplace=True)

        if self.label != "":
            # Fill missing values in y (if needed)
            if y.isnull().sum() > 0:
                if y.dtype == 'object':  
                    y.fillna(y.mode()[0], inplace=True)
                else:
                    y.fillna(y.median(), inplace=True)

        # Identify datetime columns correctly
        datetime_cols = [col for col in X.columns if np.issubdtype(X[col].dtype, np.datetime64)]

        # Convert datetime columns to numerical values (Unix timestamp)
        for col in datetime_cols:
            X[col] = X[col].astype(np.int64) // 10**9

        # One-hot encode categorical features in X
        X_encoded = pd.get_dummies(X, drop_first=True)

        if self.label != "":
            # Encode target (y)
            if y.dtype == 'object':
                if len(y.unique()) == 2:
                    y_encoded = LabelEncoder().fit_transform(y)  # Binary classification
                else:
                    y_encoded = pd.get_dummies(y, drop_first=True).values  # Multi-class classification
            else:
                y_encoded = y.values  # Regression task (numerical y)

        # Scale only original numerical features in X
        num_cols = X.select_dtypes(include=['number']).columns.difference(datetime_cols)
        scaler = StandardScaler()
        X_encoded[num_cols] = scaler.fit_transform(X_encoded[num_cols])

        base_path = "/".join(self.save_path.split("/")[:-1])

        os.makedirs(base_path, exist_ok=True)
        # Save the scaler
        with open(f"{base_path}/scaler.pkl", "wb") as f:
            pickle.dump(scaler, f)

        X_encoded.to_csv("train.csv")

        # Convert to numpy arrays
        X_res = np.array(X_encoded, dtype=np.float32)

        y_res = np.array([])

        if self.label != "":
            y_res = np.array(y_encoded, dtype=np.float32)
            print("y_res:")
            print(y_res)

        # Split the data
        if self.label != "":
            X_train, X_test, y_train, y_test = train_test_split(X_res, y_res, test_size=0.2, random_state=42)
        else:
            X_train = X_res
            X_test = X_res
            y_train = X_res
            y_test = X_res

        input_shape = X_res.shape[1]
        
        if self.task_type == 'classification' and y_res.size > 0:
            # For binary classification or multi-class classification
            output_shape = y_res.shape[1] if y_res.ndim > 1 else 1  # Handle one-hot encoding for multi-class
        elif self.task_type == 'unsupervised classification' or self.task_type == 'anomaly detection':
            output_shape = input_shape
        else:
            # For regression, output shape is the number of target values (typically 1 for single target)
            output_shape = 1

        return X_train, X_test, y_train, y_test, input_shape, output_shape

    
    def build_model(self, hidden_layers=[64, 32], activation='relu', dropout_rate=0.5):
        """
        Builds a neural network dynamically, supporting feedforward (classification/regression) and autoencoder models.

        Parameters:
        - hidden_layers (list): List of neurons per hidden layer.
        - activation (str): Activation function for hidden layers.
        - dropout_rate (float): Dropout rate for regularization.
        
        Returns:
        - Compiled Keras model.
        """
        
        if self.task_type == 'unsupervised classification' or self.task_type == 'anomaly detection':
            # Input layer
            input_layer = layers.Input(shape=(self.input_dim,))

            # Encoder
            encoded = input_layer
            for units in hidden_layers:
                encoded = layers.Dense(units, activation=activation)(encoded)
                if dropout_rate > 0:
                    encoded = layers.Dropout(dropout_rate)(encoded)

            # Latent space representation
            encoding_dim = hidden_layers[-1]  # Smallest layer is the encoded representation

            # Decoder (mirroring the encoder)
            decoded = encoded
            for units in reversed(hidden_layers[:-1]):
                decoded = layers.Dense(units, activation=activation)(decoded)
                if dropout_rate > 0:
                    decoded = layers.Dropout(dropout_rate)(decoded)

            decoded = layers.Dense(self.input_dim, activation='sigmoid')(decoded)  # Output layer for reconstruction
            loss_fn = 'mse'
            # Define autoencoder model
            model = models.Model(input_layer, decoded)

        elif self.task_type == 'classification':
            model = models.Sequential()
            model.add(layers.InputLayer(input_shape=(self.input_dim,)))

            # Add hidden layers dynamically
            for units in hidden_layers:
                model.add(layers.Dense(units, activation=activation))
                if dropout_rate > 0:
                    model.add(layers.Dropout(dropout_rate))  # Optional dropout for regularization
            
            if self.output_dim == 1:
                model.add(layers.Dense(1, activation='sigmoid'))  # Binary classification
                loss_fn = 'binary_crossentropy'
            else:
                model.add(layers.Dense(self.output_dim, activation='softmax'))  # Multi-class classification
                loss_fn = 'categorical_crossentropy'
        else:  # Regression task
            model = models.Sequential()
            model.add(layers.InputLayer(input_shape=(self.input_dim,)))

            # Add hidden layers dynamically
            for units in hidden_layers:
                model.add(layers.Dense(units, activation=activation))
                if dropout_rate > 0:
                    model.add(layers.Dropout(dropout_rate))  # Optional dropout for regularization

            model.add(layers.Dense(self.output_dim, activation='linear'))  # No activation for regression
            loss_fn = 'mse'  # Mean Squared Error for regression

        # Compile the model
        model.compile(optimizer='adam', loss=loss_fn, metrics=['accuracy'] if self.task_type == 'classification' else ['mse'])

        return model


    def fit(self, parameters, config):
        # Load parameters into the model
        if self.model is None:
            self.model = self.build_model()
            self.model.fit(self.X_train, self.y_train, epochs=1, validation_split=0.2)  # Initial training to build the model

        if parameters:
            self.model.set_weights(parameters)

        # Train the Keras model
        self.model.fit(self.X_train, self.y_train, epochs=5, validation_split=0.2)

        self.model.save(self.save_path)

        # Return updated weights
        return self.model.get_weights(), len(self.X_train), {}


    def evaluate(self, parameters, config):
        # Ensure the Keras model is initialized
        if self.model is None:
            if self.model is None:
                self.model = self.build_model()
                self.model.fit(self.X_train, self.y_train, epochs=1, validation_split=0.2)  # Initial training to build the model

        if parameters:
            self.model.set_weights(parameters)

        # Evaluate the Keras model
        loss, accuracy = self.model.evaluate(self.X_test, self.y_test, verbose=0)
        return loss, len(self.X_test), {"accuracy": accuracy}

def create_client(df, save_path, task_type, label):
    # Initialize the Flower client
    client = FederatedClient(df=df, save_path=save_path, task_type=task_type, label=label)
    return client

def sig_start_server(rounds: str, model_json: str, save_path: str, edge_server_url):
    # Define the URL of the FastAPI server
    url = f"{edge_server_url}/start_flwr_server/"

    # Define the payload (parameters) to be sent in the POST request
    data = {
        "rounds": rounds,
        "model_json": model_json,
        "save_path": save_path
    }

    # Send the POST request
    response = requests.post(url, json=data)

    # Check the response status
    if response.status_code == 200:
        print("Request successful!")
        print(f"Response: {response.json()}")
        res = response.json()
        return res["pid"]
    else:
        print(f"Request failed with status code {response.status_code}")
        print(f"Error: {response.text}")
        return None

def sig_stop_server(pid, save_path, edge_server_url):
    # Define the URL of the FastAPI server
    url = f"{edge_server_url}/stop_flwr_server/"

    # Define the payload (parameters) to be sent in the POST request
    data = {
        "pid": pid,
        "save_path": save_path
    }

    # Send the POST request
    response = requests.post(url, json=data)

    # Check the response status
    if response.status_code == 200:
        print("Request successful!")
        print(f"Response: {response.json()}")
    else:
        print(f"Request failed with status code {response.status_code}")
        print(f"Error: {response.text}")


def upload_scaler(save_path, edge_server_url):
    base_path = "/".join(save_path.split("/")[:-1])
    scaler_path = base_path + "/scaler.pkl"
    url = f"{edge_server_url}/upload_scaler/"

    with open(scaler_path, "rb") as f:
        files = {"file": (scaler_path, f, "application/octet-stream")}
        data = {"save_path": scaler_path}  # Pass the custom path

        response = requests.post(url, files=files, data=data)

    print(response.json())


# Create an ArgumentParser object
parser = argparse.ArgumentParser(description="flwr server script")

# Add named arguments
parser.add_argument("--file_path", type=str, required=True)
parser.add_argument("--save_path", type=str, required=True)
parser.add_argument("--task_type", type=str, required=True)
parser.add_argument("--label", type=str, required=True)
parser.add_argument("--rounds", type=str, required=True)
parser.add_argument("--edge_server_url", type=str, required=True)

# Parse the arguments
args = parser.parse_args()

if __name__ == "__main__":
    
    flwr_server_ip_and_port = args.edge_server_url.split("//")[-1]

    flwr_server_ip = flwr_server_ip_and_port.split(":")[0]
    flwr_server_address = flwr_server_ip + ":8080"

    df = pd.read_excel(args.file_path)

    client = create_client(df, args.save_path, args.task_type, args.label)
    upload_scaler(args.save_path, args.edge_server_url)

    model = client.build_model()
    model_json = model.to_json()

    pid = sig_start_server(args.rounds, model_json, args.save_path, args.edge_server_url)
    if pid:
        time.sleep(30)

        fl.client.start_numpy_client(server_address=flwr_server_address, client=client)

        time.sleep(10)
        sig_stop_server(pid, args.save_path, args.edge_server_url)

    