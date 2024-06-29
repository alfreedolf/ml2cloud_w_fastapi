from sklearn.metrics import fbeta_score, precision_score, recall_score
from sklearn.ensemble import RandomForestClassifier

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

def _slice_data(data, slicing_feature):
    # TODO: implement slicing
    pass

def compute_molde_metrics_on_slice(model, data, slicing_feature):
    """
    Validates the trained machine learning model using precision, recall, and F1 on slices.

    Inputs
    ------
    mode : ??? 
        Trained machine learning model.
    data: pd.DataFrame
        dataframe to be sliced
    slicing_feature : str
        name of the feature to be used to slice the data
    Returns
    -------
    precision : float
    recall : float
    fbeta : float
    """
    # slice data
    X, y = _slice_data(data, slicing_feature)

    # do inference using the model
    preds = inference(model, X)

    compute_model_metrics(y, preds) 
    fbeta = fbeta_score(y, preds, beta=1, zero_division=1)
    precision = precision_score(y, preds, zero_division=1)
    recall = recall_score(y, preds, zero_division=1)
    return precision, recall, fbeta


def inference(model, X):
    """ Run model inferences and return the predictions.

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
    predictions = model.predict(X)
    return predictions
