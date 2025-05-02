import mlflow
from mlflow.tracking import MlflowClient
import subprocess
from inference import infer
import os

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


def train_fn(df, save_path, task_type, labels, rounds, edge_server_url): 
    # Save DataFrame to CSV
    os.makedirs("tmp", exist_ok=True)
    tmp_path = "tmp/data.xlsx"
    df.to_excel(tmp_path, index=False)
    
    command = [
        "python",
        "flwr_client.py",  # Replace with your actual script path
        "--file_path", str(tmp_path),
        "--save_path", str(save_path),
        "--task_type", str(task_type),
        "--rounds", str(rounds),
        "--edge_server_url", str(edge_server_url),
    ]

    if len(labels) > 0:
        command.append("--labels")

    # appending labels to command
    for label in labels:
        command.append(label)
    
    # Start the external Python script as a subprocess
    process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    stdout, stderr = process.communicate()

    os.remove(tmp_path)
    
    print(stderr)

    return "success"