import mlflow
# from iris_pipeline import load_dataset, train_and_log_model, inference, params, accuracy, X_train

from sklearn.datasets import load_diabetes
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split

# TODO: Setup MLflow.

# Set our tracking server uri for logging
mlflow.set_tracking_uri(uri="http://127.0.0.1:8080")

# Create a new MLflow Experiment
mlflow.set_experiment("MLflow Quickstart")

# TODO: Setup autologging.
mlflow.autolog()
# Start an MLflow run
# with mlflow.start_run():
#     # Log the hyperparameters
#     # Define the model hyperparameters
#     params = {
#         "solver": "lbfgs",
#         "max_iter": 1000,
#         "multi_class": "auto",
#         "random_state": 8888,
#     }
    
#     mlflow.log_params(params)

#     # Log the loss metric
#     mlflow.log_metric("accuracy", accuracy)

#     # Set a tag that we can use to remind ourselves what this run was for
#     mlflow.set_tag("Training Info", "Basic LR model for iris data")

#     # Infer the model signature
#     signature = inference(X_train, model.predict(X_train))

#     # Log the model
#     rf = 
#         model_info = mlflow.sklearn.log_model(
#         sk_model=,
#         artifact_path="iris_model",
#         signature=signature,
#         input_example=X_train,
#         registered_model_name="tracking-quickstart",
#     )

db = load_diabetes()

X_train, X_test, y_train, y_test = train_test_split(db.data, db.target)

# Create and train models
rf = RandomForestRegressor(n_estimators=100, max_depth=6, max_features=3)
rf.fit(X_train, y_train)

# Use the model to make predictions on the test dataset
predictions = rf.predict(X_test)
