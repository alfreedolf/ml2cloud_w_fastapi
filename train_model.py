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
from ml.model import compute_model_metrics, compute_model_metrics_on_slice, train_model

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
def train(model_output_file_name: str):
    model = train_model(X_train, y_train)
    joblib.dump(model, os.path.join(os.getcwd(), "models", model_output_file_name+".joblib"))
    return model


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--model-output-file-name", type=str, default="model")
    parser.add_argument("--test-split-ratio", type=float, default=0.20)
    parser.add_argument("--input_data_file", type=str, default="census_clean.csv")
    parser.add_argument("--k-fold-splits", type=int, default=-1)
    parser.add_argument("--slice-by", type=str, default=None)
    parser.add_argument("--slice-class", type=str, default=None)
    args = parser.parse_args()

    # Add code to load in the data.
    census_df = pd.read_csv(os.path.join(os.getcwd(), "data", args.input_data_file))

    if args.k_fold_splits < 2:
        # Optional enhancement, use K-fold cross validation instead of a train-test split.
        train_data, test_data = train_test_split(census_df, test_size=args.test_split_ratio, random_state=42)
        X_train, y_train, encoder, lb = process_data(
            train_data,
            categorical_features=CATEGORICAL_FEATURES,
            label="salary",
            training=True,
        )
        model = train(args.model_output_file_name)
        X_test, y_test, _, _ = process_data(
            test_data,
            categorical_features=CATEGORICAL_FEATURES,
            label="salary",
            slice_by=args.slice_by,
            slice_class=args.slice_class,
            training=False,
            encoder=encoder,
            lb=lb,
        )
        if args.slice_by is None:
            # scoring the model

            predictions = model.predict(X_test)
            accuracy = accuracy_score(y_test, predictions)
            precision, recall, fbeta = compute_model_metrics(y_test, predictions)
            print(f"model accuracy: {accuracy}")
        else:
            precision, recall, fbeta = compute_model_metrics_on_slice(
                model=model,
                categorical_features=CATEGORICAL_FEATURES,
                data=census_df,
                label="salary",
                feature_slice=args.slice_by,
                slice_class=args.slice_class,
                encoder=encoder,
                label_binarizer=lb,
            )
            

        print(f"model precision: {precision} | model recall = {recall} | model fbeta = {fbeta}")
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
                test_data,
                categorical_features=CATEGORICAL_FEATURES,
                label="salary",
                slice_by=args.slice_by,
                slice_class=args.slice_class,
                training=False,
                encoder=encoder,
                lb=lb,
            )
            predictions = model.predict(X_test)
            # score = accuracy_score(y_test, predictions)
            precision, recall, fbeta = compute_model_metrics(y_test, predictions)
            scores.append([precision, recall, fbeta])

        cum_scores = [sum(element) for element in scores]
        max_cum_score = max(cum_scores)
        # get the best model
        best_index = cum_scores.index(max_cum_score)
        best_model = models[best_index]
        predictions = best_model.predict(X_test)
        accuracy = accuracy_score(y_test, predictions)

        if args.slice_by is None:
            precision, recall, fbeta = compute_model_metrics(y_test, predictions)
            print(f"max score in {args.k_fold_splits} training: {scores[best_index]}")
        else:
            precision, recall, fbeta = compute_model_metrics_on_slice(model=model,
                categorical_features=CATEGORICAL_FEATURES,
                data=census_df,
                label="salary",
                feature_slice=args.slice_by,
                slice_class=args.slice_class,
                encoder=encoder,
                label_binarizer=lb
                )
        print(f"max score in {args.k_fold_splits} training, on slice {args.slice_class}: {scores[best_index]}")
        precision, recall, fbeta = compute_model_metrics(y_test, predictions)
        
        print(f"max score in {args.k_fold_splits} training: {scores[best_index]}")
        joblib.dump(best_model, os.path.join(os.getcwd(), "models", f"{args.model_output_file_name}.joblib"))
