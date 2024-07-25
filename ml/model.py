import os
import logging
import joblib
from sklearn.metrics import fbeta_score, precision_score, recall_score
from sklearn.ensemble import RandomForestClassifier

from ml.data import process_data

logging.basicConfig(level=logging.INFO)


# Optional: implement hyperparameter tuning.
def train_model(X_train, y_train):
    """
    Trains a machine learning model and returns it.

    Inputs
    ------
    X_train : np.array
        Training data.
    y_train : np.array
        Labels.
    Returns
    -------
    model
        Trained machine learning model.
    """

    rfc = RandomForestClassifier()
    rfc.fit(X_train, y_train)
    return rfc


def compute_model_metrics(y, preds):
    """
    Validates the trained machine learning model using precision, recall, and F1.

    Inputs
    ------
    y : np.array
        Known labels, binarized.
    preds : np.array
        Predicted labels, binarized.
    Returns
    -------
    precision : float
    recall : float
    fbeta : float
    """
    fbeta = fbeta_score(y, preds, beta=1, zero_division=1)
    precision = precision_score(y, preds, zero_division=1)
    recall = recall_score(y, preds, zero_division=1)
    return precision, recall, fbeta


def compute_model_metrics_on_slice(
    model, data, categorical_features, label, feature_slice, slice_class, encoder, label_binarizer, metrics_file_path="slice_output.txt"
):
    """
    Validates the trained machine learning model using precision, recall, and F1 on slices.

    Inputs
    ------
    mode : ???
        Trained machine learning model.
    data: pd.DataFrame
        dataframe to be sliced
    categorical_features: list
        categorical features
    label: str
        label to be predicted
    feature_slice:
        name of the feature to be used to slice the data
    slice_class:
        slice class chosen
    encoder:
        encoder to be used, reuse it from training phase
    label_binarizer:
        label binarizer to be used, reuse it from training phase
    metrics_file_path:
        path where to write performances values
    Returns
    -------
    precision : float
    recall : float
    fbeta : float
    """

    # preprocess data
    X, y, _, _ = process_data(
        data,
        categorical_features=categorical_features,
        label=label,
        slice_by=feature_slice,
        slice_class=slice_class,
        training=False,
        encoder=encoder,
        lb=label_binarizer,
    )

    # do inference using the model
    preds = inference(X=X, model=model)

    compute_model_metrics(y, preds)
    fbeta = fbeta_score(y, preds, beta=1, zero_division=1)
    precision = precision_score(y, preds, zero_division=1)
    recall = recall_score(y, preds, zero_division=1)
    
    try:
        with open(metrics_file_path, 'w') as metrics_file:
            metrics_file.write(f"precision: {precision}\n")
            metrics_file.write(f"recall: {recall}\n")  
            metrics_file.write(f"fbeta: {fbeta}\n")
    except FileNotFoundError as fnfe:
        logging.error("File %s not found", metrics_file_path)
    return precision, recall, fbeta


def _load_model_from_path(model_path: str):
    """Loads a model from joblib file
    Inputs
    ------
    model_path : str
        Trained machine learning model path.
    Returns
    -----
        unpickled model
    """
    with open(model_path, mode="rb") as model_file:
        model = joblib.load(model_file)
        return model


def inference(X, model="default"):
    """Run model inferences and return the predictions.

    Inputs
    ------
    model : ???
        Trained machine learning model.
    X : np.array
        Data used for prediction.
    Returns
    -------
    preds : np.array
        Predictions from the model.
    """
    # in case a string is provided as path
    if isinstance(model, str):
        if model == "default" or model == "test":
            model = _load_model_from_path(os.path.join("models", "model.joblib"))
        else:
            model = _load_model_from_path(os.path.join("..", "models", model))
    
    # in case nor a string or a RandomForest Classifier is provided as input
    elif not isinstance(model, RandomForestClassifier):
        logging.error("wrong argument fro model parameter, no inference will be given")
        return None

    predictions = model.predict(X)
    return predictions
