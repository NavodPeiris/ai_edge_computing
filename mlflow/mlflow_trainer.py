import argparse
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import tensorflow as tf
from tensorflow.keras import layers, models
import mlflow
import mlflow.tensorflow
import os
import pickle

def load_and_preprocess_data(df, label=None, save_path=None):
    """Preprocess data for model training and save preprocessing artifacts"""
    # Separate features and target
    if label and label in df.columns:
        X = df.drop(columns=[label])
        y = df[label]
    else:
        X = df
        y = None

    # Fill missing values in X
    for col in X.columns:
        if X[col].isnull().sum() > 0:
            if X[col].dtype == 'object':  # Categorical feature
                X[col].fillna(X[col].mode()[0], inplace=True)
            else:  # Numerical feature
                X[col].fillna(X[col].median(), inplace=True)

    # Fill missing values in y if it exists
    if y is not None and y.isnull().sum() > 0:
        if y.dtype == 'object':
            y.fillna(y.mode()[0], inplace=True)
        else:
            y.fillna(y.median(), inplace=True)

    # Identify datetime columns
    datetime_cols = [col for col in X.columns if np.issubdtype(X[col].dtype, np.datetime64)]

    # Convert datetime columns to numerical values
    for col in datetime_cols:
        X[col] = X[col].astype(np.int64) // 10**9

    # One-hot encode categorical features
    X_encoded = pd.get_dummies(X, drop_first=True)

    # Scale numerical features
    num_cols = X.select_dtypes(include=['number']).columns.difference(datetime_cols)
    scaler = StandardScaler()
    X_encoded[num_cols] = scaler.fit_transform(X_encoded[num_cols])

    # Save scaler if path provided
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        scaler_path = os.path.join(os.path.dirname(save_path), "scaler.pkl")
        with open(scaler_path, "wb") as f:
            pickle.dump(scaler, f)
        print(f"Saved scaler to {scaler_path}")

    # Convert to numpy arrays
    X_res = np.array(X_encoded, dtype=np.float32)
    
    if y is not None:
        # Encode target if needed
        if y.dtype == 'object':
            if len(y.unique()) == 2:
                from sklearn.preprocessing import LabelEncoder
                y_encoded = LabelEncoder().fit_transform(y)
            else:
                y_encoded = pd.get_dummies(y, drop_first=True).values
        else:
            y_encoded = y.values
        
        y_res = np.array(y_encoded, dtype=np.float32)
        return X_res, y_res, X_encoded.shape[1]
    else:
        return X_res, None, X_encoded.shape[1]

def build_model(input_dim, task_type, output_dim=None):
    """Build neural network model based on task type"""
    hidden_layers = [64, 32]
    activation = 'relu'
    dropout_rate = 0.5
    
    if task_type in ['unsupervised classification', 'anomaly detection']:
        # Input layer
        input_layer = layers.Input(shape=(input_dim,))

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

        decoded = layers.Dense(input_dim, activation='sigmoid')(decoded)  # Output layer for reconstruction
        loss_fn = 'mse'
        # Define autoencoder model
        model = models.Model(input_layer, decoded)
        model.compile(optimizer='adam', loss=loss_fn)

    elif task_type == 'classification':
        model = models.Sequential()
        model.add(layers.InputLayer(input_shape=(input_dim,)))

        # Add hidden layers dynamically
        for units in hidden_layers:
            model.add(layers.Dense(units, activation=activation))
            if dropout_rate > 0:
                model.add(layers.Dropout(dropout_rate))  # Optional dropout for regularization
        
        if output_dim == 1 or output_dim is None:
            model.add(layers.Dense(1, activation='sigmoid'))  # Binary classification
            loss_fn = 'binary_crossentropy'
        else:
            model.add(layers.Dense(output_dim, activation='softmax'))  # Multi-class classification
            loss_fn = 'categorical_crossentropy'
        model.compile(optimizer='adam', loss=loss_fn, metrics=['accuracy'])
    else:  # Regression task
        model = models.Sequential()
        model.add(layers.InputLayer(input_shape=(input_dim,)))

        # Add hidden layers dynamically
        for units in hidden_layers:
            model.add(layers.Dense(units, activation=activation))
            if dropout_rate > 0:
                model.add(layers.Dropout(dropout_rate))  # Optional dropout for regularization

        model.add(layers.Dense(output_dim if output_dim else 1, activation='linear'))  # No activation for regression
        loss_fn = 'mse'  # Mean Squared Error for regression
        model.compile(optimizer='adam', loss=loss_fn, metrics=['mse'])

    return model

def train_model(model, X_train, y_train, X_val, y_val, epochs=10, batch_size=32):
    """Train the model and return history"""
    if y_train is not None:
        history = model.fit(
            X_train, y_train,
            epochs=epochs,
            batch_size=batch_size,
            validation_data=(X_val, y_val),
            verbose=1
        )
    else:
        # For autoencoders, X is both input and output
        history = model.fit(
            X_train, X_train,
            epochs=epochs,
            batch_size=batch_size,
            validation_data=(X_val, X_val),
            verbose=1
        )
    
    return history

def save_model_artifacts(model, save_path):
    """Save model in keras format"""
    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    # Save in keras format
    model.save(save_path)
    print(f"Saved model to {save_path}")

def register_model(model, model_name, task_type, description=""):
    """Register model to MLflow Model Registry"""
    # Log model to MLflow
    mlflow.tensorflow.log_model(model, "model")
    
    # Set model description and task type as tags
    mlflow.set_tag("task_type", task_type)
    mlflow.set_tag("description", description)
    
    # Register model in MLflow Model Registry
    result = mlflow.register_model(f"runs:/{mlflow.active_run().info.run_id}/model", model_name)
    return result

def main():
    parser = argparse.ArgumentParser(description="Train and register a model with MLflow")
    parser.add_argument("--file_path", required=True, help="Path to data file (CSV or Excel)")
    parser.add_argument("--model_name", required=True, help="Name for the registered model")
    parser.add_argument("--task_type", required=True, choices=[
        "classification", "regression", "unsupervised classification", "anomaly detection"
    ], help="Type of ML task")
    parser.add_argument("--save_path", required=True, help="Path to save model artifacts")
    parser.add_argument("--label", help="Name of the target column (if applicable)")
    parser.add_argument("--description", default="", help="Model description")
    parser.add_argument("--epochs", type=int, default=10, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size for training")
    parser.add_argument("--mlflow_tracking_uri", default="http://localhost:5001", 
                       help="URI for MLflow tracking server")
    
    args = parser.parse_args()
    
    # Set MLflow tracking URI
    mlflow.set_tracking_uri(args.mlflow_tracking_uri)

    # Set experiment
    mlflow.set_experiment(args.model_name)
    
    # Load data
    if args.file_path.endswith('.xlsx') or args.file_path.endswith('.xls'):
        df = pd.read_excel(args.file_path)
    else:
        df = pd.read_csv(args.file_path)
    
    # Preprocess data and save scaler
    if args.task_type in ["classification", "regression"] and args.label:
        X, y, input_dim = load_and_preprocess_data(df, args.label, args.save_path)
        output_dim = 1
        if y is not None and y.ndim > 1:
            output_dim = y.shape[1]
    else:
        X, _, input_dim = load_and_preprocess_data(df, save_path=args.save_path)
        y = None
        output_dim = None
    
    # Split data
    if y is not None:
        X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
    else:
        X_train, X_val = train_test_split(X, test_size=0.2, random_state=42)
        y_train = y_val = None
    
    # Start MLflow run
    with mlflow.start_run(run_name=args.model_name):
        # Build model
        model = build_model(input_dim, args.task_type, output_dim)
        
        # Log parameters
        mlflow.log_param("task_type", args.task_type)
        mlflow.log_param("input_dim", input_dim)
        mlflow.log_param("epochs", args.epochs)
        mlflow.log_param("batch_size", args.batch_size)
        
        # Train model
        history = train_model(model, X_train, y_train, X_val, y_val, 
                             epochs=args.epochs, batch_size=args.batch_size)
        
        # Log metrics
        for epoch, (loss, val_loss) in enumerate(zip(history.history['loss'], history.history['val_loss'])):
            mlflow.log_metric("loss", loss, step=epoch)
            mlflow.log_metric("val_loss", val_loss, step=epoch)
        
        # Save model artifacts locally
        save_model_artifacts(model, args.save_path)
        scaler_path = os.path.join(os.path.dirname(args.save_path), "scaler.pkl")
        mlflow.log_artifact(scaler_path)  # Log the scaler as an artifact
        
        # Register model to MLflow
        result = register_model(model, args.model_name, args.task_type, args.description)
        
        print(f"Model {args.model_name} registered successfully with version {result.version}!")


if __name__ == "__main__":
    main()