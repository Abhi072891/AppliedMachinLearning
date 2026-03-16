# app.py
import os
import warnings
import joblib
from flask import Flask, request, render_template, jsonify

warnings.filterwarnings("ignore")

from score import score as score_func

app = Flask(__name__)

# Load model (svc.pkl) at startup. Ensure svc.pkl is present in working dir.
try:
    model = joblib.load("svc.pkl")
except Exception as e:
    model = None
    app.logger.warning("svc.pkl not found or failed to load: %s", e)

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/score", methods=["POST"])
def score_route():
    """
    Accepts form-encoded POST (field name 'Text') or JSON {'text': ...}
    Returns rendered html (score.html). Tests parse the template by id.
    """
    # Accept either form 'Text' or JSON 'text'
    text = request.form.get("Text")
    if text is None:
        json_data = request.get_json(silent=True) or {}
        text = json_data.get("text", "")

    if model is None:
        return "Model not available on server", 500

    # Use threshold=0.5 by default (adjustable)
    pred_bool, propensity = score_func(text, model, threshold=0.5)

    label = "Spam" if pred_bool else "Not Spam"

    # Render the simple template with IDs used in tests
    return render_template("score.html", text=text, pred=label, prob=round(propensity, 6))

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    # Bind to 0.0.0.0 in Docker so host can reach it
    app.run(host="0.0.0.0", port=port, debug=False)