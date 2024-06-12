import sys
import os
import pytest
import pandas as pd
print(os.getcwd())
sys.path.append("./ml")

TEST_DATA_PATH="data"


@pytest.fixture(name="training_data")
def training_data():
    df_training_data = pd.read_csv(os.path.join(TEST_DATA_PATH, "census_clean.csv"))
    yield df_training_data
    
@pytest.fixture(name="X_train")
def training_features(training_data):
    yield training_data.drop(columns=["salary"], inplace=False)
    
@pytest.fixture(name="y_train")
def training_labels(training_data):
    y_train = training_data["salary"]
    print(y_train)
    yield y_train
    
    