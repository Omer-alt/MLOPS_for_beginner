import pickle
import asyncio
import os
from dotenv import load_dotenv
from sklearn.datasets import load_iris
from pydantic import BaseModel, Field, confloat
from sklearn.utils import Bunch
import httpx
from typing import List

from typing import Union

from contextlib import asynccontextmanager

from fastapi import FastAPI

ml_models = {}

load_dotenv()
LOGISTIC_MODEL = os.getenv('LOGISTIC_MODEL')
RF_MODEL = os.getenv('RF_MODEL') 

# Data format
# class IrisData(BaseModel):
#     sepal_length: float
#     sepal_width: float
#     petal_length: float
#     petal_width: float
    
# Enhanced schema validation with constraints and descriptions
class IrisData(BaseModel):
    sepal_length: float = Field(ge=1.0, le=10.0, description="Sepal width in cm", example=3.5)
    sepal_width: float = Field(ge=1.0, le=10.0, description="Sepal width in cm", example=3.5)
    petal_length: float = Field(ge=1.0, le=10.0, description="Petal length in cm", example=1.4)
    petal_width: float = Field(ge=0.0, le=10.0, description="Petal width in cm", example=0.2)

    
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
    ml_models["rf_model"] = model_loader(RF_MODEL)
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
    await asyncio.sleep(2)  # Simulates a delay of seconds
    
    predicted_class = model.predict(input_data)[0]
    predicted_class_name = load_iris().target_names[predicted_class]

    return IrisPrediction(
        predicted_class=predicted_class, predicted_class_name=predicted_class_name
    )

    
# Testing the asynchronous behavior with multiple requests
async def send_request(client, model_name, data):
    response = await client.post(f"http://127.0.0.1:8000/predict/{model_name}", json=data)
    print("Single responses", response.json(), data, f"http://127.0.0.1:8000/predict/{model_name}")
    return response.json()

@app.get("/test_concurrent")
async def test_concurrent():
    
    data = {
        "sepal_length": 5.1,
        "sepal_width": 3.5,
        "petal_length": 1.4,
        "petal_width": 0.2
    }
    async with httpx.AsyncClient() as client:
        
        # Send multiple requests concurrently
        tasks = [
            send_request(client, "logreg_model", data),
            send_request(client, "rf_model", data)
        ]
        
        # Gather results from all requests
        results = await asyncio.gather(*tasks)
        
    return {"responses": results}



@app.post("/test_concurrent")
async def test_concurrent(models: List[str], iris_data: List[IrisData]):

    # Running predictions in parallel for each model
    tasks = [ predict(models_name, i_data) for models_name, i_data in zip(models, iris_data)]
    results = await asyncio.gather(*tasks)
    
    return {"responses": results}

"""_summary_: Data to test /test_concurrent route
{
    "models": ["logreg_model", "rf_model", "rf_model"],
    "iris_data": [{
        "sepal_length": 5.1,
        "sepal_width": 3.5,
        "petal_length": 1.4,
        "petal_width": 0.2
    }, {
        "sepal_length": 7.2,
        "sepal_width": 3.2, 
        "petal_length": 6.0, 
        "petal_width": 1.8
    },
    {
        "sepal_length": 6.3, 
        "sepal_width": 2.3, 
        "petal_length": 4.4, 
        "petal_width": 1.3
    }]
}
"""