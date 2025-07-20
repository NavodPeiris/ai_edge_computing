import flwr as fl
import tensorflow as tf
import os
import argparse
import psutil
from tensorflow.keras.models import model_from_json


class CustomFedAvg(fl.server.strategy.FedAvg):
    def __init__(self, save_dir, model_json, num_rounds, **kwargs):
        super().__init__(**kwargs)  # Pass other FedAvg parameters via kwargs

        self.save_dir = "/".join(save_dir.split("/")[:-1])
        self.global_model = model_from_json(model_json)  # Initialize the model architecture
        self.num_rounds = num_rounds  # Save total rounds to compare in eval

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
    
    def aggregate_evaluate(self, server_round, results, failures):
        aggregated_evaluation = super().aggregate_evaluate(server_round, results, failures)

        return aggregated_evaluation


# Create an ArgumentParser object
parser = argparse.ArgumentParser(description="flwr server script")

# Add named arguments
parser.add_argument("--rounds", type=str, required=True, help="number of rounds to train")
parser.add_argument("--model_json", type=str, required=True)
parser.add_argument("--save_path", type=str, required=True)
parser.add_argument("--num_clients", type=str, required=True)
parser.add_argument("--port", type=str, required=True)

# Parse the arguments
args = parser.parse_args()

if __name__ == "__main__":

    # Start the server with the custom strategy
    strategy = CustomFedAvg(
        save_dir=args.save_path,
        model_json = args.model_json,
        num_rounds = args.rounds,
        fraction_fit=1.0,         # Use all available clients for training
        fraction_evaluate=1.0,    # Use all available clients for evaluation
        min_fit_clients=int(args.num_clients),        # Minimum clients required for training
        min_evaluate_clients=int(args.num_clients),   # Minimum clients required for evaluation
        min_available_clients=int(args.num_clients),  # Minimum clients required to start a round
    )

    hist = fl.server.start_server(
        server_address=f"0.0.0.0:{args.port}",  # Adjust the address as needed
        config=fl.server.ServerConfig(num_rounds=int(args.rounds)),  # Number of federated learning rounds
        strategy=strategy,
    )