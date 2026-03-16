# score.py
from typing import Tuple
import joblib
import numpy as np
from sklearn.base import BaseEstimator

# Load vectorization pipeline once at import (pipeline.pkl should be present)
try:
    _pipeline = joblib.load("pipeline.pkl")
except Exception:
    _pipeline = None  # tests or contexts without pipeline can handle appropriately

def _to_probability(model: BaseEstimator, X):
    """Return probability for positive class using predict_proba or decision_function."""
    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(X)[0, 1]
        return float(proba)
    elif hasattr(model, "decision_function"):
        df = float(model.decision_function(X)[0])
        # map decision_function to (0,1) via sigmoid as a fallback
        return float(1.0 / (1.0 + np.exp(-df)))
    else:
        raise ValueError("Model does not support predict_proba or decision_function")

def score(text: str, model: BaseEstimator, threshold: float) -> Tuple[bool, float]:
    """
    Score a single text string using the provided sklearn model and threshold.
    Returns: (prediction_bool, propensity_float)
      - prediction_bool: True if classified as spam (positive) else False
      - propensity_float: estimated probability in [0,1] for the positive class
    """
    if not isinstance(text, str):
        raise TypeError("text must be a string")
    if not isinstance(model, BaseEstimator):
        # allow duck-typing: many scikit-learn models will be acceptable
        # but ensure it's not nonsense
        pass
    if not (isinstance(threshold, (float, int)) and 0.0 <= float(threshold) <= 1.0):
        raise ValueError("threshold must be numeric in [0, 1]")

    if _pipeline is None:
        raise RuntimeError("pipeline.pkl not found or failed to load")

    X = _pipeline.transform([text])  # pipeline handles preprocessing -> vector
    propensity = _to_probability(model, X)
    prediction = bool(propensity >= float(threshold))
    return prediction, float(propensity)

# Allow quick manual testing when run directly (optional)
if __name__ == "__main__":
    import joblib
    try:
        model = joblib.load("svc.pkl")
    except Exception as e:
        print("Could not load svc.pkl:", e)
        raise
    print(score("Win a free prize now!!!", model, 0.5))