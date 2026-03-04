import pickle
import warnings
from flask import Flask, request, render_template

warnings.filterwarnings("ignore")
app = Flask(__name__)

# Load model and pipeline (assumed saved by train.ipynb)
pipeline = pickle.load(open('pipeline.pkl', 'rb'))
model = pickle.load(open('svc.pkl', 'rb'))

# Use the score function for classification
from score import score as score_func

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/score', methods=['POST'])
def score_route():
    text = request.form.get('text', '')
    # Get binary prediction and probability
    prediction_bool, propensity = score_func(text, model, threshold=0.5)
    # Convert boolean to a human-readable label
    label = 'Spam' if prediction_bool else 'Not Spam'
    # Render results using HTML template
    return render_template('score.html', pred=label, prob=propensity, text=text)

if __name__ == '__main__':
    app.run()