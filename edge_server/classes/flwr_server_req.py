from typing import List, Any
from pydantic import BaseModel

class FlwrRequestParams(BaseModel):
    rounds: str
    model_json: str  # The model architecture in JSON format
    save_path: str
    num_clients: str

class TrainParams(BaseModel):
    pid: str
    port: str
    task_type: str
    epochs: str
    rounds: str
    edge_server_url: str
    model_json: str
    hidden_layers: List[int]
    labels: List[str]
    metrics: List[str]
    loss_fn: str

class FetchTrainParams(BaseModel):
    port: str

class FlwrStopParams(BaseModel):
    pid: int
    save_path: str

class DeliverModel(BaseModel):
    model_path: str

class DeliverScaler(BaseModel):
    scaler_path: str

class DeliverEncoder(BaseModel):
    encoder_path: str

class ModelMetadata(BaseModel):
    name: str
    description: str
    model_folder: str
    task: str

class DeliverModelJson(BaseModel):
    models: List[ModelMetadata]

class CheckModelAvailable(BaseModel):
    model_path: str

class DeleteModel(BaseModel):
    model_path: str