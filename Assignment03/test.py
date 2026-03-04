import pickle
import subprocess
import time
import re
import pytest
import requests

from score import score

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

# Setup dummy pipeline and model for testing
pipeline = TfidfVectorizer()
train_texts = ["buy now", "hello friend", "free money", "see you soon"]
train_labels = [1, 0, 1, 0]
X_train = pipeline.fit_transform(train_texts)
model = LogisticRegression(random_state=0)
model.fit(X_train, train_labels)
# Save pipeline and model as if retrained
with open('pipeline.pkl', 'wb') as f:
    pickle.dump(pipeline, f)
with open('svc.pkl', 'wb') as f:
    pickle.dump(model, f)

def test_smoke_score():
    """Smoke test: score() should return (bool, float) without error."""
    pred, prop = score("test message", model, threshold=0.5)
    assert isinstance(pred, bool)
    assert isinstance(prop, float)
    assert 0.0 <= prop <= 1.0

def test_propensity_range():
    """Propensity (probability) should be between 0 and 1."""
    _, prop = score("special offer", model, threshold=0.5)
    assert 0.0 <= prop <= 1.0

def test_threshold_edges():
    """Edge thresholds: threshold=0 should always predict True if any probability,
    threshold=1 should predict True only if probability == 1."""
    text = "free free free"
    pred0, _ = score(text, model, threshold=0.0)
    assert pred0 == True  # any non-zero probability >= 0
    pred1, prob1 = score(text, model, threshold=1.0)
    assert prob1 < 1.0 and pred1 == False

def test_obvious_spam_ham():
    """Checks obvious spam vs. ham behavior."""
    spam_text = "win lottery free money now"
    ham_text = "see you at the meeting"
    pred_spam, prop_spam = score(spam_text, model, threshold=0.5)
    pred_ham, prop_ham = score(ham_text, model, threshold=0.5)
    assert pred_spam == True
    assert pred_ham == False
    assert prop_spam >= prop_ham

def test_invalid_threshold():
    """Thresholds outside [0,1] should raise ValueError."""
    with pytest.raises(ValueError):
        score("any", model, threshold=-0.1)
    with pytest.raises(ValueError):
        score("any", model, threshold=1.1)

def test_flask_integration():
    """Integration test: run Flask app, POST to /score, and validate HTML response."""
    # Start the Flask app as a subprocess
    proc = subprocess.Popen(["python", "app.py"], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    time.sleep(5)  # allow server to start
    try:
        res = requests.post("http://127.0.0.1:5000/score", data={"text": "hello world"})
        assert res.status_code == 200
        content = res.text
        # Check that expected fields appear in HTML
        assert "Score" in content
        assert "Propensity" in content or re.search(r"\d+\.\d+", content)
    finally:
        proc.terminate()
        proc.wait(timeout=5)