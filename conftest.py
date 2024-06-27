import sys
import os
import pytest
import pandas as pd
from sklearn.model_selection import train_test_split
from ml.data import process_data

from ml.model import train_model

print(os.getcwd())
sys.path.append("./ml")

TEST_DATA_PATH = "data"


@pytest.fixture(name="training_data")
def training_data():
    df_training_data = pd.read_csv(os.path.join(TEST_DATA_PATH, "census_clean.csv"))
    yield df_training_data


@pytest.fixture(name="preprocessed_data")
def preprocess_data(training_data, categorical_features):
    X, y, encoder, lb = process_data(
        X=training_data,
        categorical_features=categorical_features,
        label="salary",
        training=True,
    )
    yield {"X": X, "y": y, "encoder": encoder, "lb": lb}


@pytest.fixture(name="preprocessed_split_data")
def split_data(preprocessed_data, random_state=42):
    # Optional enhancement, use K-fold cross validation instead of a train-test split.
    train_data_x, test_data_x, train_data_y, test_data_y = train_test_split(
        preprocessed_data["X"],
        preprocessed_data["y"],
        test_size=0.20,
        random_state=random_state,
    )
    return {
        "X_train": train_data_x,
        "X_test": test_data_x,
        "y_train": train_data_y,
        "y_test": test_data_y,
    }


@pytest.fixture(name="X_train")
def training_features(preprocessed_data):
    yield preprocessed_data["X"]


@pytest.fixture(name="y_train")
def training_labels(preprocessed_data):
    print(preprocessed_data["y"])
    yield preprocessed_data["y"]


@pytest.fixture(name="y_test")
def testing_labels(preprocessed_data):
    print(preprocessed_data["y"])
    yield preprocessed_data["y"]


@pytest.fixture(name="categorical_features")
def categorical_features():
    features = [
        "workclass",
        "education",
        "marital-status",
        "occupation",
        "relationship",
        "race",
        "sex",
        "native-country",
    ]
    yield features

@pytest.fixture(name="fitted_model")
def fit_model(preprocessed_split_data):
    trained_model = train_model(
        X_train=preprocessed_split_data["X_train"],
        y_train=preprocessed_split_data["y_train"],
    )
    yield trained_model

@pytest.fixture(name="predictions")
def predictions(preprocessed_split_data):
    trained_model = train_model(
        X_train=preprocessed_split_data["X_train"],
        y_train=preprocessed_split_data["y_train"],
    )
    predictions = trained_model.predict(preprocessed_split_data["X_test"])
    return predictions

