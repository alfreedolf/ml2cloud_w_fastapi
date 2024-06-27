import pytest
import sys
import os

# print(os.getcwd())
from ml.model import compute_model_metrics, train_model, inference


def test_train_model(X_train, y_train):
    trained_model = train_model(X_train=X_train, y_train=y_train)
    assert trained_model is not None


def test_inference(fitted_model, preprocessed_split_data):
    predictions = inference(model=fitted_model, X=preprocessed_split_data["X_test"])
    assert predictions is not None
    assert predictions.shape == preprocessed_split_data["y_test"].shape


def test_compute_model_metrics(preprocessed_split_data, predictions):
    # Optional enhancement, use K-fold cross validation instead of a train-test split.
    compute_model_metrics(preprocessed_split_data["y_test"], preds=predictions)


if __name__ == "__main__":
    # The `-v` flag is for verbose mode
    pytest_args = ["-v"]
    pytest.main(pytest_args)
