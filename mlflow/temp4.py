# Load model directly
import mlflow
from mlflow.models.signature import infer_signature, set_signature
from transformers import pipeline
from PIL import Image
from mlflow.models.signature import ModelSignature
from mlflow.types.schema import Schema, ColSpec

pipe = pipeline("image-segmentation", model="facebook/detr-resnet-50-panoptic")


mlflow.set_tracking_uri("http://localhost:5001")
mlflow.set_experiment("detr-resnet-50-panoptic")
# Set model description and task type as description
mlflow.set_tag("task_type", "image-segmentation")
mlflow.set_tag("description", "this model can segment images")


# Define the input schema (file + string)
input_schema = Schema([
    ColSpec(type="binary"),   # or 'file', but MLflow understands this as 'binary'
])

# Define output schema (infer or set manually)
image = Image.open("depth.jpeg").convert("RGB")

# Example output to infer output schema
output = pipe(image)

from mlflow.models.signature import infer_signature
output_schema = infer_signature(output).outputs

# Construct signature
signature = ModelSignature(inputs=input_schema, outputs=output_schema)



model_info = mlflow.transformers.log_model(
    transformers_model=pipe, 
    artifact_path='detr-resnet-50-panoptic',
    task='image-segmentation',
    signature=signature,
)

mlflow.register_model(f"runs:/{mlflow.active_run().info.run_id}/model", "detr-resnet-50-panoptic")
