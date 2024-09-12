import pytest

from day1.simple_ml_pipeline import IrisDataset, Model

def test_load_dataset():
    iris_dataset = IrisDataset()
    df = iris_dataset.load_dataset()
    assert not df.empty, "The DataFrame should not be empty after loading the dataset."
    assert "species" in df.columns, "The DataFrame should have a 'species' column."

def test_model_accuracy():
    # Load dataset
    iris_dataset = IrisDataset()
    df = iris_dataset.load_dataset()
    
    # Train model
    model = Model()
    # X_train, X_test, y_train, y_test = model.train(df)
    _, X_test, _, y_test = model.train(df)
    accuracy = model.get_accuracy(X_test, y_test)
    
    # Assert accuracy is greater than 80%
    assert accuracy > 0.8, f"Model accuracy is below 80%, got {accuracy:.2f}"
    
def test_displays_rows_positive_number():
    iris_dataset = IrisDataset()
    df = iris_dataset.load_dataset()

    # Test to verify that the number of lines is positive
    with pytest.raises(ValueError, match="The number of lines must be a positive integer."):
        iris_dataset.displays_rows(df, number_rows=-3)



