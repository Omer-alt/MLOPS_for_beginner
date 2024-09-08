from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split


def load_dataset():
    iris = load_iris()
    X_train, X_test, y_train, y_test = train_test_split(
        iris.data, iris.target, test_size=0.2, random_state=42
    )
    return X_train, X_test, y_train, y_test


def train_and_log_model(params):
    X_train, X_test, y_train, y_test = load_dataset()

    model = LogisticRegression(**params)

    model.fit(X_train, y_train)
    # Predict on the test set
    accuracy = inference(model, X_test, y_test)

    # TODO: Log the hyperparameters

    # TODO: Log the loss metric

    # TODO: Set a tag that we can use to remind ourselves what this run was for

    # TODO: Infer the model signature to be used when saving model

    # TODO: Save the model, make sure to provide the signature and an input example.

    return model


def inference(model, X_test, y_test):
    predictions = model.predict(X_test)
    accuracy = accuracy_score(y_test, predictions)
    print(f"Accuracy of the model is: {accuracy}.")

    return accuracy


if __name__ == "__main__":
    # TODO: Set up MLflow.

    # Define the model hyperparameters
    params = {
        "solver": "lbfgs",
        "max_iter": 1000,
        "multi_class": "auto",
        "random_state": 8888,
    }
    model = train_and_log_model(params)

    X_train, X_test, y_train, y_test = load_dataset()

    # TODO: Load the model back from MLflow for more predictions
    # model = ...

    accuracy = inference(model, X_test, y_test)
