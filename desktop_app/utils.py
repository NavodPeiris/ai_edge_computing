import mlflow
from mlflow.tracking import MlflowClient
import subprocess
from inference import infer
import os
import streamlit as st
import pandas as pd
import json
import requests

edge_server_url = "http://127.0.0.1:8001"

# URL of the Grafana dashboard or panel (make sure it's publicly accessible or authenticated)
grafana_url = "http://localhost:3003"

# MLflow server address
mlflow_tracking_uri = "http://127.0.0.1:5001"

# initializing MLflow client
mlflow.set_tracking_uri(mlflow_tracking_uri)
client = MlflowClient()

def inf(df, labels, model_path, task_type, edge_server_url): 
    res = infer(df, labels, task_type, model_path, edge_server_url)
    return res


def train_fn(user_id, df, save_path, task_type, labels, rounds, edge_server_url, hidden_layers, epochs, initializer, num_clients=None, port=None, model_json=None, metrics=None, loss_fn=None, showPortMsg=None): 
    # Save DataFrame to CSV
    os.makedirs(f"tmp/{user_id}", exist_ok=True)

    tmp_path = f"tmp/{user_id}/data.xlsx"
    df.to_excel(tmp_path, index=False)

    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"
    
    command = [
        "python",
        "-u",
        "flwr_client.py",  # Replace with your actual script path
        "--file_path", str(tmp_path),
        "--save_path", str(save_path),
        "--task_type", str(task_type),
        "--epochs", str(epochs),
        "--rounds", str(rounds),
        "--edge_server_url", str(edge_server_url)
    ]

    if len(labels) > 0:
        command.append("--labels")

    # appending labels to command
    for label in labels:
        command.append(label)

    if len(hidden_layers) == 0:
        hidden_layers = [64, 32]

    command.append("--hidden_layers")

    # appending layers
    for layer in hidden_layers:
        command.append(str(layer))

    if not initializer:
        command.append("--initializer")
        command.append("False")
        command.append("--port")
        command.append(port)
        command.append("--model_json")
        command.append(model_json)
        command.append("--loss_fn")
        command.append(loss_fn)

        command.append("--metrics")
        for metric in metrics:
            command.append(metric)
    else:
        command.append("--initializer")
        command.append("True")
        command.append("--num_clients")
        command.append(num_clients)
    
    final_accuracy = None
    final_loss = None

    process = subprocess.Popen(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,  # merge stderr into stdout
        text=True,
        bufsize=1,  # line buffered
        env=env
    )

    try:
        # Read output line-by-line
        for line in process.stdout:
            print(line.strip())  # optional: show live output

            if line.startswith("Final Evaluation Accuracy:"):
                final_accuracy = round(float(line.split(":")[-1]) * 100, 2)

            if line.startswith("Final Evaluation Loss:"):
                final_loss = round(float(line.split(":")[-1]) * 100, 2)

            if line.startswith("Port Fed Server Running On:") and int(num_clients) > 1:
                fed_server_port = int(line.split(":")[-1])
                showPortMsg(f"Share This Port With Other Clients: {fed_server_port}")

    finally:
        process.stdout.close()
        process.wait()  # wait for the process to finish
        os.remove(tmp_path)

    return final_accuracy, final_loss