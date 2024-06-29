# Script to train machine learning model.

import logging
import os
import joblib
import argparse
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split, KFold

# Add the necessary imports for the starter code.
import pandas as pd

from ml.data import process_data
from ml.model import train_model

logging.basicConfig(level=logging.INFO)


# set categorical data
CATEGORICAL_FEATURES = [
    "workclass",
    "education",
    "marital-status",
    "occupation",
    "relationship",
    "race",
    "sex",
    "native-country",
]

# Train and save a model.
def train():
    model = train_model(X_train, y_train)
    joblib.dump(model, os.path.join(os.getcwd(), "models", "model.pkl"))
    return model


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--input_data_file", type=str, default="census_clean.csv")
    parser.add_argument("--k-fold-splits", type=int, default=-1)
    args = parser.parse_args()

    

    # Add code to load in the data.
    census_df = pd.read_csv(os.path.join(os.getcwd(), "data", args.input_data_file))

    if args.k_fold_splits < 2:
        # Optional enhancement, use K-fold cross validation instead of a train-test split.
        train_data, test_data = train_test_split(census_df, test_size=0.20, random_state=42)
        X_train, y_train, encoder, lb = process_data(
            train_data, categorical_features=CATEGORICAL_FEATURES, label="salary", training=True
        )
        model = train()
        # scoring the model
        X_test, y_test, _, _ = process_data(
            test_data, categorical_features=CATEGORICAL_FEATURES, label="salary", training=False, encoder=encoder, lb=lb
        )
        predictions = model.predict(X_test)
        score = accuracy_score(y_test, predictions)
        print(f"model accuracy score: {score}")
    else:
        kf = KFold(n_splits=args.k_fold_splits)
        models = []
        scores = []
        for i, (train_index, test_index) in enumerate(kf.split(census_df)):
            # logging.info("Fold %d:", i)
            # logging.info("  Train: index = %s:", train_index)
            # logging.info("  Test:  index = %s:", test_index)


            train_data = census_df.iloc[train_index]
            test_data = census_df.iloc[test_index]
            X_train, y_train, encoder, lb = process_data(
                train_data, categorical_features=CATEGORICAL_FEATURES, label="salary", training=True
            )

            model = train_model(X_train, y_train)
            models.append(model)

            # scoring the model
            X_test, y_test, _, _ = process_data(
                test_data, categorical_features=CATEGORICAL_FEATURES, label="salary", training=False, encoder=encoder, lb=lb
            )        
            predictions = model.predict(X_test)
            score = accuracy_score(y_test, predictions)
            scores.append(score)
        # get the best model
        best_index = scores.index(max(scores))
        best_model = models[best_index]
        print(f"max accuracy score in k-fold training: {max(scores)}")
        joblib.dump(best_model, os.path.join(os.getcwd(), "models", f"model_{args.k_fold_splits}-fold.pkl"))
