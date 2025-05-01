# Load model directly
import mlflow
from mlflow.models.signature import infer_signature, set_signature
from transformers import pipeline
from PIL import Image
from mlflow.models.signature import ModelSignature
from mlflow.types.schema import Schema, ColSpec

pipe = pipeline("text-classification", model="tabularisai/multilingual-sentiment-analysis")


mlflow.set_tracking_uri("http://localhost:5001")
mlflow.set_experiment("multilingual-sentiment-analysis")
# Set model description and task type as description
mlflow.set_tag("task_type", "text-classification")
mlflow.set_tag("description", "this model can sentiment analysis in multiple languages")


# Define the input schema (file + string)
input_schema = Schema([
    ColSpec(type="string", name="text"),   # or 'file', but MLflow understands this as 'binary'
])

# Define output schema (infer or set manually)

# Example output to infer output schema
output = pipe("hello i am sad")

from mlflow.models.signature import infer_signature
output_schema = infer_signature(output[0]).outputs

# Construct signature
signature = ModelSignature(inputs=input_schema, outputs=output_schema)



model_info = mlflow.transformers.log_model(
    transformers_model=pipe, 
    artifact_path='multilingual-sentiment-analysis',
    task='text-classification',
    signature=signature,
)

mlflow.register_model(f"runs:/{mlflow.active_run().info.run_id}/model", "multilingual-sentiment-analysis")
