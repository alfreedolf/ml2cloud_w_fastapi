# Script to train machine learning model.

import os
from sklearn.model_selection import train_test_split

# Add the necessary imports for the starter code.
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import LabelBinarizer

# Add code to load in the data.
census_df = pd.read_csv(os.path.join("..", "data", "census.csv"))

# Proces the test data with the process_data function.
def process_data(train_data:pd.DataFrame, categorical_features:list, label:str, training:bool):
    X_train = train_data[categorical_features]
    y_train = train_data[label]
    encoder = LabelEncoder()
    lb = LabelBinarizer()
    
    if training:
        X_train = X_train.apply(encoder.fit_transform)
        y_train = lb.fit_transform(y_train)
    return X_train, y_train, encoder, lb


# Optional enhancement, use K-fold cross validation instead of a train-test split.
train, test = train_test_split(census_df, test_size=0.20)

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
    train, categorical_features=cat_features, label="salary", training=True
)


# Train and save a model.
def train(train_data:pd.DataFrame):
    pass