from pydantic import BaseModel

class FlwrRequestParams(BaseModel):
    rounds: str
    model_json: str  # The model architecture in JSON format
    save_path: str

class FlwrStopParams(BaseModel):
    pid: int
    save_path: str

class DeliverModel(BaseModel):
    model_path: str

class DeliverScaler(BaseModel):
    scaler_path: str
    