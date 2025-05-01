import mlflow
import transformers
from mlflow.models.signature import infer_signature, set_signature

# Load a pre-trained model from Hugging Face Hub
model = transformers.AutoModelForSequenceClassification.from_pretrained('bert-base-uncased')
tokenizer = transformers.AutoTokenizer.from_pretrained('bert-base-uncased')

mlflow.set_tracking_uri("http://localhost:5001")
mlflow.set_experiment("bert-base-uncased")
# Set model description and task type as description
mlflow.set_tag("task_type", "text-classification")
mlflow.set_tag("description", "this model classify texts")


model_info = mlflow.transformers.log_model(
    transformers_model={'model': model, 'tokenizer': tokenizer},
    artifact_path='bert-base-uncased',
    task='text-classification'
)

input_data = "Hello, how are you?"

mlflow.register_model(f"runs:/{mlflow.active_run().info.run_id}/model", "bert-base-uncased")

model = mlflow.transformers.load_model(f"runs:/{mlflow.active_run().info.run_id}/bert-base-uncased")

signature = infer_signature(input_data, model.predict(input_data))

set_signature(model_info.model_uri, signature)