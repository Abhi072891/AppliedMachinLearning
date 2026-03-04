import pickle
from typing import Tuple
from sklearn.base import BaseEstimator
import numpy as np

def score(text: str, model: BaseEstimator, threshold: float) -> Tuple[bool, float]:
    """
    Compute spam/ham prediction and propensity for input text.

    Args:
        text (str): The input text to classify.
        model (BaseEstimator): A trained sklearn classifier with predict_proba or decision_function.
        threshold (float): Decision threshold between 0 and 1 for classifying as spam.

    Returns:
        Tuple[bool, float]: (prediction, propensity) where prediction is True if spam and False otherwise,
        and propensity is the probability estimate of the 'spam' class.
    """
    # Validate threshold range
    if not (0.0 <= threshold <= 1.0):
        raise ValueError("Threshold must be between 0 and 1")
    # Load pre-fitted TF-IDF pipeline
    pipeline = pickle.load(open('pipeline.pkl', 'rb'))
    # Transform the input text into feature vector (pipeline expects iterable of texts)
    X = pipeline.transform([text])
    # Get probability estimate for the positive class (spam)
    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(X)[0, 1]
    elif hasattr(model, "decision_function"):
        # Convert decision function output to [0,1] via sigmoid if no probabilities
        df_val = model.decision_function(X)[0]
        proba = 1.0 / (1.0 + np.exp(-df_val))
    else:
        raise ValueError("Model does not support probability or decision function")
    # Determine binary prediction based on threshold
    prediction = bool(proba >= threshold)
    return prediction, float(proba)


