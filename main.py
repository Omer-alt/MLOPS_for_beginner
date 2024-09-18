import pandas as pd
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

from fastapi import FastAPI, BackgroundTasks

from fastapi.responses import HTMLResponse
from evidently.report import Report
from evidently.metric_preset import DataDriftPreset, DataQualityPreset

# Global variable to log requests
DATA_LOG = []
WINDOW_SIZE = 24
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
    
    
# Load the Iris dataset as the reference data
def load_dataset():
    iris = load_iris()
    df = pd.DataFrame(iris.data, columns=iris.feature_names)
    df["species"] = iris.target
    df["species_name"] = df.apply(
        lambda x: str(iris.target_names[int(x["species"])]), axis=1
    )
    return df

# Load the production data from DATA_LOG
def load_production_data(window_size: int = WINDOW_SIZE):
    global DATA_LOG
    # Convert the last `window_size` records to a DataFrame
    
    # df = pd.DataFrame(DATA_LOG[-window_size:], columns=['sepal length (cm)',	'sepal width (cm)',	'petal length (cm)',	'petal width (cm)',	'species',	'species_name' ])
    if (window_size <= len(DATA_LOG)) or (window_size >= len(DATA_LOG) and len(DATA_LOG)!=0):
        df = pd.DataFrame(DATA_LOG[-window_size:], columns=['sepal_length', 'sepal_width', 'petal_length', 'petal_width', 'species', 'species_name'])
        df.columns = ['sepal length (cm)', 'sepal width (cm)', 'petal length (cm)', 'petal width (cm)', 'species', 'species_name']
        print(df.head())
        return df
    

    
    


# Create a Dashboard with Evidently
def generate_drift_report():
    reference_data = load_dataset()
    
    print(reference_data.head())
    production_data = load_production_data()

    # Create Evidently report comparing reference and production data
    report = Report(metrics=[
        DataDriftPreset(),
        DataQualityPreset()
    ])
    report.run(reference_data=reference_data, current_data=production_data)
    
    return report

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
    # await asyncio.sleep(1)  # Simulates a delay of seconds
    
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



@app.post("/batch_predict")
async def test_concurrent(models: List[str], iris_data: List[IrisData], background_tasks: BackgroundTasks):

    # Running predictions in parallel for each model
    tasks = [ predict(models_name, i_data) for models_name, i_data in zip(models, iris_data)]
    
    results = await asyncio.gather(*tasks)
    
    # Add it inside the DATA_LOG (Global variable)
    for input_pred in zip(iris_data, results):
        print(input_pred)
        prediction_result = input_pred[1].dict()
        background_tasks.add_task(log_data, {**input_pred[0].dict(), "species_name": prediction_result["predicted_class_name"], "species": prediction_result["predicted_class"] })
    
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


# Day 6 Ajustments

# Function to log feature and prediction data to DATA_LOG
def log_data(data: dict):
    global DATA_LOG
    DATA_LOG.append(data)
    
# Prediction endpoint that logs requests in the background
@app.post('/predict')
async def predict_with_background(input_data: IrisData, background_tasks: BackgroundTasks):
    # Simulate prediction process
    # prediction = {"prediction": "setosa"}
    prediction = await predict("rf_model", input_data)

    # Log the request and prediction using background tasks
    background_tasks.add_task(log_data, {**input_data.dict(), "species_name": prediction.dict()["predicted_class_name"], "species": prediction.dict()["predicted_class"] })

    return prediction

@app.get('/monitoring')
async def monitoring():
    try:
        # Generate the drift report
        report = generate_drift_report()

        # Save report as HTML
        report_html_path = "drift_report.html"
        report.save_html(report_html_path)

        # Read the HTML file and return as response
        with open(report_html_path, "r") as f:
            html_content = f.read()

        return HTMLResponse(content=html_content)
    except Exception as e:
        return {"error": str(e)}


@app.get("/logs")
def get_logs():
    return DATA_LOG



