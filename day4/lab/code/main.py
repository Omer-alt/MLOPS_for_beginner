import pickle
import asyncio
import os
from dotenv import load_dotenv
from sklearn.datasets import load_iris
from pydantic import BaseModel
from sklearn.utils import Bunch

from typing import Union

from contextlib import asynccontextmanager

from fastapi import FastAPI

ml_models = {}

load_dotenv()
LOGISTIC_MODEL = os.getenv('LOGISTIC_MODEL')
RF_MODEL = os.getenv('RF_MODEL') 

# Data format
class IrisData(BaseModel):
    sepal_length: float
    sepal_width: float
    petal_length: float
    petal_width: float
    
class IrisPrediction(BaseModel):
    predicted_class: int
    predicted_class_name: str


def model_loader(filename):
    loaded_model = pickle.load(open(filename, 'rb'))
    
    return loaded_model

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Load the ML model
    # ml_models["logreg_model"] = model_loader('logreg_model.sav')
    # ml_models["rm_model"] = model_loader('rf_model.sav')
    ml_models["logreg_model"] = model_loader(LOGISTIC_MODEL)
    ml_models["rm_model"] = model_loader(RF_MODEL)
    yield
    
    # Clean up the ML models and release the resources
    ml_models.clear()

# app = FastAPI()
app = FastAPI(lifespan=lifespan)


@app.get("/")
def read_root():
    return "Hello World!"

@app.get("/health")
def get_heathstatus():
    return {"status": "up and running" }

@app.get("/models")
async def get_models():
   print("model", ml_models)
   return {"availables_models": list(ml_models.keys()) }

@app.post("/predict/{model_name}")
async def predict(models_name, iris_data: IrisData):

    input_data =[[iris_data.sepal_length, iris_data.sepal_width, iris_data.petal_length, iris_data.petal_width]]
    
    
    # logreg_model = ml_models["logreg_model"]
    model = ml_models[models_name]
    
    # Simulate a long-running operation with asyncio.sleep
    await asyncio.sleep(10)  # Simulates a delay of seconds
    
    predicted_class = model.predict(input_data)[0]
    predicted_class_name = load_iris().target_names[predicted_class]

    return IrisPrediction(
        predicted_class=predicted_class, predicted_class_name=predicted_class_name
    )

    
    
@app.get("/items/{item_id}")
def read_item(item_id: int, q: Union[str, None] = None):
    return {"item_id": item_id, "q": q}