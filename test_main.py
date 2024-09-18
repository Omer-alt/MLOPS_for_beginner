import pytest
import pandas as pd
from unittest.mock import MagicMock
from fastapi.testclient import TestClient
from main import WINDOW_SIZE, app, load_dataset, load_production_data, ml_models  

# Use TestClient to create a test client for the FastAPI app
client = TestClient(app)

def test_read_root():
    response = client.get("/")
    assert response.status_code == 200
    assert response.json() == "Hello World!"

def test_health_status():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "up and running"}

def test_get_models():
    response = client.get("/models")
    assert response.status_code == 200
    assert "availables_models" in response.json()
    # assert response.json() == {"availables_models": ["logreg_model", "rf_model"]}
    
    
#  Test the loading of dataset
def test_load_dataset():
    df = load_dataset()
    assert isinstance(df, pd.DataFrame), "The output should be a pandas DataFrame."
    assert not df.empty, "The DataFrame should not be empty after loading the dataset."
    
def test_load_dataset():
    df = load_production_data()
    
    # Test if the output is a pandas DataFrame
    assert isinstance(df, pd.DataFrame), "The output should be a pandas DataFrame."
    
    # Test if the DataFrame has the correct number of rows
    assert len(df) <= WINDOW_SIZE, f"The DataFrame should contain at most {WINDOW_SIZE} rows, but got {len(df)}."

    
    
    
    
    
    
    
    
    
    
    
    
    
    
    # # Call the lifespan event to trigger model loading
    # with client:
    #     response = client.get("/models")
    #     assert response.status_code == 200
    #     assert "availables_models" in response.json()
    #     assert response.json()["availables_models"] == ["logreg_model", "rf_model"]

# @pytest.mark.asyncio
# async def test_predict_logreg(mocked_model_loader):
#     # Mock the logistic model for testing
#     mock_model = mocked_model_loader.return_value
#     mock_model.predict.return_value = [0]  # Mock the prediction to always return 0

#     data = {
#         "sepal_length": 5.1,
#         "sepal_width": 3.5,
#         "petal_length": 1.4,
#         "petal_width": 0.2
#     }

#     response = client.post("/predict/logreg_model", json=data)
#     assert response.status_code == 200
#     assert response.json()["predicted_class_name"] == "setosa"  # Assuming class 0 is 'setosa'

# @pytest.mark.asyncio
# async def test_concurrent_predictions(mocked_model_loader):
#     # Mock both models (logreg_model and rf_model) for testing
#     mock_logreg_model = mocked_model_loader.return_value
#     mock_rf_model = mocked_model_loader.return_value

#     mock_logreg_model.predict.return_value = [0]
#     mock_rf_model.predict.return_value = [1]

#     data = {
#         "sepal_length": 5.1,
#         "sepal_width": 3.5,
#         "petal_length": 1.4,
#         "petal_width": 0.2
#     }

#     response = client.get("/test_concurrent")
#     assert response.status_code == 200
#     assert len(response.json()["responses"]) == 2
