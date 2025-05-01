import mlflow

mlflow.set_tracking_uri("http://localhost:5001")

model = mlflow.transformers.load_model(f"runs:/f13c4241413642458a7278928d7b3157/bert-base-uncased")

res = model.predict("Hello, how are you?")
print(res)