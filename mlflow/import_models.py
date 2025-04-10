# import_models.py
import os
import argparse
import mlflow
import mlflow.tensorflow
from tensorflow.keras.models import load_model
import json
import pickle
import glob

def import_model_to_registry(model_path, model_name, task_type, description, artifacts_dir=None):
    """Import a pre-trained model into the MLflow registry"""
    try:
        # Load the model
        model = load_model(model_path)
        
        # Start an MLflow run
        with mlflow.start_run(run_name=f"import-{model_name}"):
            # Log model parameters
            model_params = {
                "input_shape": model.input_shape[1:] if hasattr(model, "input_shape") else None,
                "output_shape": model.output_shape[1:] if hasattr(model, "output_shape") else None,
                "layers": len(model.layers) if hasattr(model, "layers") else None
            }
            
            for param_name, param_value in model_params.items():
                if param_value is not None:
                    mlflow.log_param(param_name, param_value)
            
            # Log model to MLflow
            mlflow.tensorflow.log_model(model, "model")
            
            # Set model description and task type as tags
            mlflow.set_tag("task_type", task_type)
            mlflow.set_tag("description", description)
            
            # Log additional artifacts if provided
            if artifacts_dir and os.path.exists(artifacts_dir):
                # Find all pickle files (likely scalers, encoders, etc.)
                artifact_files = glob.glob(os.path.join(artifacts_dir, "*.pkl"))
                for artifact_file in artifact_files:
                    mlflow.log_artifact(artifact_file)
                    print(f"Logged artifact: {os.path.basename(artifact_file)}")
            
            # Register model in MLflow Model Registry
            result = mlflow.register_model(f"runs:/{mlflow.active_run().info.run_id}/model", model_name)
            
            print(f"Model {model_name} successfully registered with version {result.version}")
            return True
    except Exception as e:
        print(f"Error importing model: {e}")
        return False

def main():
    parser = argparse.ArgumentParser(description="Import models to MLflow registry")
    parser.add_argument("--model_path", required=True, help="Path to the saved model file (.keras, .h5, or SavedModel directory)")
    parser.add_argument("--model_name", required=True, help="Name for the registered model")
    parser.add_argument("--task_type", required=True, 
                        choices=["classification", "regression", "unsupervised classification", "anomaly detection"],
                        help="Type of ML task")
    parser.add_argument("--description", default="Imported model", help="Model description")
    parser.add_argument("--artifacts_dir", help="Directory containing additional model artifacts (e.g., scalers)")
    parser.add_argument("--mlflow_tracking_uri", default="http://localhost:5001", help="MLflow tracking server URI")
    
    args = parser.parse_args()
    
    # Set MLflow tracking URI
    mlflow.set_tracking_uri(args.mlflow_tracking_uri)

    # Set experiment
    mlflow.set_experiment(args.model_name)
    
    # Import the model
    success = import_model_to_registry(
        args.model_path, 
        args.model_name,
        args.task_type,
        args.description,
        args.artifacts_dir
    )
    
    if success:
        print(f"Model {args.model_name} successfully imported to MLflow registry")
    else:
        print("Failed to import model")

if __name__ == "__main__":
    main()