# Script to train machine learning model.

import os
import joblib
from sklearn.model_selection import train_test_split

# Add the necessary imports for the starter code.
import pandas as pd

from ml.data import process_data
from ml.model import train_model


# Train and save a model.
def train():
    model = train_model(X_train, y_train)
    joblib.dump(model, os.path.join(os.getcwd(), "models", "model.pkl"))


if __name__ == "__main__":
    # Add code to load in the data.
    census_df = pd.read_csv(os.path.join(os.getcwd(), "data", "census_clean.csv"))

    # Optional enhancement, use K-fold cross validation instead of a train-test split.
    train_data, test_data = train_test_split(census_df, test_size=0.20, random_state=42)

    cat_features = [
        "workclass",
        "education",
        "marital-status",
        "occupation",
        "relationship",
        "race",
        "sex",
        "native-country",
    ]
    X_train, y_train, encoder, lb = process_data(
        train_data, categorical_features=cat_features, label="salary", training=True
    )
    train()
