import sys
import os
import pytest
import pandas as pd
from ml.data import process_data
print(os.getcwd())
sys.path.append("./ml")

TEST_DATA_PATH="data"


@pytest.fixture(name="training_data")
def training_data():
    df_training_data = pd.read_csv(os.path.join(TEST_DATA_PATH, "census_clean.csv"))
    yield df_training_data

@pytest.fixture(name="preprocessed_data")
def preprocess_data(training_data, categorical_features):
    X, y, encoder, lb = process_data(X=training_data, categorical_features=categorical_features, label="salary", training=True)
    yield {"X": X, "y": y, "encoder": encoder, "lb": lb}

@pytest.fixture(name="X_train")
def training_features(preprocessed_data):
    yield preprocessed_data["X"]
    
@pytest.fixture(name="y_train")
def training_labels(preprocessed_data):
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
    
    