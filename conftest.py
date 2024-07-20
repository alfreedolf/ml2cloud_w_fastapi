import sys
import os
import pytest
import pandas as pd
from sklearn.model_selection import train_test_split
from ml.data import process_data

from ml.model import train_model

import random

print(os.getcwd())
sys.path.append("./ml")

TEST_DATA_PATH = "data"
SAMPLE_SIZE = 50


@pytest.fixture(name="training_data")
def training_data():
    df_training_data = pd.read_csv(os.path.join(TEST_DATA_PATH, "census_clean.csv"))
    df_training_data = df_training_data.rename(columns=lambda x: x.strip())
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


def _generate_value(training_data, field_name: str):
    """
    Generates a random full name from training data.
    """
    sample_values = random.sample(training_data[field_name].values.tolist(), SAMPLE_SIZE)
    value = random.choice(sample_values)
    if isinstance(value, str):
        value.replace("\n", "")
        value.strip()
    return value



@pytest.fixture(name="census_record")
def census_record(training_data):
    age: str = _generate_value(training_data, field_name="age")
    workclass: str = _generate_value(training_data, field_name="workclass")
    fnlgt: int = int(_generate_value(training_data, field_name="fnlgt"))
    education: str = _generate_value(training_data, field_name="education")
    education_num: int = int(_generate_value(training_data, field_name="education-num"))
    marital_status: str = _generate_value(training_data, field_name="marital-status")
    occupation: str = _generate_value(training_data, field_name="occupation")
    relationship: str = _generate_value(training_data, field_name="relationship")
    race: str = _generate_value(training_data, field_name="race")
    sex: str = _generate_value(training_data, field_name="sex")
    capital_gain: int = int(_generate_value(training_data, field_name="capital-gain"))
    capital_loss: int = int(_generate_value(training_data, field_name="capital-loss"))
    hours_per_week: int = int(_generate_value(training_data, field_name="hours-per-week"))
    native_country: str = _generate_value(training_data, field_name="native-country")

    record = f"""age: {age}, workclass: {workclass}, fnlgt: {fnlgt}, education: {education}, education-num: {education_num}, marital-status: {marital_status}, occupation: {occupation}, relationship: {relationship}, race: {race}, sex: {sex}, capital-gain: {capital_gain}, capital-loss: {capital_loss}, hours-per-week: {hours_per_week}, hours-per-week: {native_country}"""

    return record
