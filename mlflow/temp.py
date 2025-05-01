# Load model directly
import mlflow
from mlflow.models.signature import infer_signature, set_signature
from transformers import pipeline
from PIL import Image
from mlflow.models.signature import ModelSignature
from mlflow.types.schema import Schema, ColSpec

pipe = pipeline("document-question-answering", model="naver-clova-ix/donut-base-finetuned-docvqa")


mlflow.set_tracking_uri("http://localhost:5001")
mlflow.set_experiment("donut-base-finetuned-docvqa")
# Set model description and task type as description
mlflow.set_tag("task_type", "document-question-answering")
mlflow.set_tag("description", "this model answer questions based on documents")


# Define the input schema (file + string)
input_schema = Schema([
    ColSpec(type="binary", name="image"),   # or 'file', but MLflow understands this as 'binary'
    ColSpec(type="string", name="question")
])

# Define output schema (infer or set manually)

# Example output to infer output schema
example_image = Image.open("receipt.jpg").convert("RGB")
output = pipe(example_image, "what is the total price?")
from mlflow.models.signature import infer_signature
output_schema = infer_signature(output).outputs

# Construct signature
signature = ModelSignature(inputs=input_schema, outputs=output_schema)



model_info = mlflow.transformers.log_model(
    transformers_model=pipe, 
    artifact_path='donut-base-finetuned-docvqa',
    task='document-question-answering',
    signature=signature,
)

mlflow.register_model(f"runs:/{mlflow.active_run().info.run_id}/model", "donut-base-finetuned-docvqa")
