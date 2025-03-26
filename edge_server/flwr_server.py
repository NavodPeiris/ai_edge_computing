import flwr as fl
import tensorflow as tf
import os
import argparse
import psutil
from tensorflow.keras.models import model_from_json

def find_and_terminate_process_on_port(port):
    # Iterate through all running processes
    for proc in psutil.process_iter(['pid', 'name']):
        try:
            # Get the connections of the process
            for conn in proc.connections(kind='inet'):
                # Check if the connection is using the specified port
                if conn.laddr.port == port:
                    print(f"Found process on port {port}: {proc.info}")
                    # Terminate the process
                    proc.terminate()
                    print(f"Terminated process with PID: {proc.info['pid']}")
                    return
        except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
            # Skip processes that no longer exist or that we don't have permission to access
            pass
    print(f"No process found on port {port}")


class CustomFedAvg(fl.server.strategy.FedAvg):
    def __init__(self, save_dir, model_json, **kwargs):
        super().__init__(**kwargs)  # Pass other FedAvg parameters via kwargs

        self.save_dir = "/".join(save_dir.split("/")[:-1])
        self.global_model = model_from_json(model_json)  # Initialize the model architecture

    def aggregate_fit(self, server_round, results, failures):
        # Call the parent method to aggregate weights
        aggregated_result = super().aggregate_fit(server_round, results, failures)

        # The aggregated_result is a tuple: (Parameters, dict)
        if aggregated_result is not None:
            aggregated_parameters, _ = aggregated_result  # Unpack the tuple

            # Save the global model after aggregation
            print(f"Saving global model for round {server_round}...")
            aggregated_weights = fl.common.parameters_to_ndarrays(aggregated_parameters)
            self.global_model.set_weights(aggregated_weights)
            os.makedirs(self.save_dir, exist_ok=True)
            self.global_model.save(os.path.join(self.save_dir, f"model_round_{server_round}.h5"))

        return aggregated_result



# Create an ArgumentParser object
parser = argparse.ArgumentParser(description="flwr server script")

# Add named arguments
parser.add_argument("--rounds", type=str, required=True, help="number of rounds to train")
parser.add_argument("--model_json", type=str, required=True)
parser.add_argument("--save_path", type=str, required=True)

# Parse the arguments
args = parser.parse_args()

if __name__ == "__main__":

    # Start the server with the custom strategy
    strategy = CustomFedAvg(
        save_dir=args.save_path,
        model_json = args.model_json,
        fraction_fit=1.0,         # Use all available clients for training
        fraction_evaluate=1.0,    # Use all available clients for evaluation
        min_fit_clients=1,        # Minimum clients required for training
        min_evaluate_clients=1,   # Minimum clients required for evaluation
        min_available_clients=1,  # Minimum clients required to start a round
    )

    find_and_terminate_process_on_port(8080)

    fl.server.start_server(
        server_address="0.0.0.0:8080",  # Adjust the address as needed
        config=fl.server.ServerConfig(num_rounds=int(args.rounds)),  # Number of federated learning rounds
        strategy=strategy,
    )