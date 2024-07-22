import logging
from os.path import join
import numpy as np
import pandas as pd
from joblib import dump, load
from sklearn.preprocessing import LabelBinarizer, OneHotEncoder
from pathlib import Path


def slice_data(data: pd.DataFrame, feature: str, feature_value: str) -> pd.DataFrame:
    """
    Slices the data respect to a feature.

    Inputs
    ------
    data : pandas.DataFrame
        Input data
    feature : str
        feature to be used to slice the data
    Returns
    -------
    sliced_data : pd.DataFrame
    """
    slice_of_data = data[data[feature] == feature_value]
    return slice_of_data


def process_data(
    X,
    categorical_features=[],
    label=None,
    slice_by=None,
    slice_class=None,
    training=True,
    encoder=None,
    lb=None,
    serialized_encoder='default',
    serialized_lb=join("data", "lb.joblib"),
):
    """Process the data used in the machine learning pipeline.

    Processes the data using one hot encoding for the categorical features and a
    label binarizer for the labels. This can be used in either training or
    inference/validation.

    Note: depending on the type of model used, you may want to add in functionality that
    scales the continuous data.

    Inputs
    ------
    X : pd.DataFrame
        Dataframe containing the features and label. Columns in `categorical_features`
    categorical_features: list[str]
        List containing the names of the categorical features (default=[])
    label : str
        Name of the label column in `X`. If None, then an empty array will be returned
        for y (default=None)
    slice_by : str
        Name of the feature to be used to slice the data, if None, no slicing will be done.
    slice_class : str
        Value of the feature to be used to get the slice. Only used if slice_by is not None.
    training : bool
        Indicator if training mode or inference/validation mode.
    encoder : sklearn.preprocessing._encoders.OneHotEncoder
        Trained sklearn OneHotEncoder, only used if training=False.
    lb : sklearn.preprocessing._label.LabelBinarizer
        Trained sklearn LabelBinarizer, only used if training=False.

    Returns
    -------
    X : np.array
        Processed data.
    y : np.array
        Processed labels if labeled=True, otherwise empty np.array.
    encoder : sklearn.preprocessing._encoders.OneHotEncoder
        Trained OneHotEncoder if training is True, otherwise returns the encoder passed
        in.
    lb : sklearn.preprocessing._label.LabelBinarizer
        Trained LabelBinarizer if training is True, otherwise returns the binarizer
        passed in.
    """

    if label is not None:
        if slice_by is not None and training is False:
            X = slice_data(X, slice_by, slice_class)
        y = X[label]
        X = X.drop([label], axis=1)
    else:
        y = np.array([])

    X_categorical = X[categorical_features].values
    X_continuous = X.drop(*[categorical_features], axis=1)

    if training is True:
        # initializing encoder
        encoder = OneHotEncoder(sparse_output=False, handle_unknown="ignore")
        

        lb = LabelBinarizer()
        
        X_categorical = encoder.fit_transform(X_categorical)
        # serializing encoder
        dump(encoder, serialized_encoder)
        y = lb.fit_transform(y.values).ravel()
        # serializing label binarizer
        dump(lb, serialized_lb)        
    else:
        if encoder is None:
            try:
                if serialized_encoder == 'default':
                    encoder_path = join("..", "data", "ohe.joblib")
                elif serialized_encoder == 'test':
                    encoder_path = join("data", "ohe.joblib")
                else:
                    encoder_path = join("..", "data", serialized_encoder)
                encoder = load(encoder_path)
            except FileNotFoundError:
                logging.error("Encoder File %s not found", encoder)
        # if lb is None:
        #     try:
        #         if serialized_lb == 'default':
        #             lb_path = join("..", "data", "lb.joblib")
        #         elif serialized_lb == 'test':
        #             lb_path = join("data", "lb.joblib")
        #         else:
        #             lb_path = join("..", "data", serialized_lb)
        #         lb = load(lb_path)
        #     except FileNotFoundError:
        #         logging.error("Label Binarizer File %s not found", lb)
        try:
            X_categorical = encoder.transform(X_categorical)
        except AttributeError as transform_error:
            logging.error("Attribute error in encoding categorical data: %s", transform_error)

        try:
            y = lb.transform(y.values).ravel()
        # Catch the case where y is None because we're doing inference.
        except AttributeError:
            logging.error("Attribute error, we are doing inference, so we have no label values")

    X = np.concatenate([X_continuous, X_categorical], axis=1)
    return X, y, encoder, lb
