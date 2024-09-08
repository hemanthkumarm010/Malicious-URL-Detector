from flask import Flask, render_template, request, jsonify
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np

app = Flask(__name__)

# Load the model and vectorizer
model = joblib.load('url_model.pkl')
vectorizer = joblib.load('vectorizer.pkl')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()  # Get JSON data from POST request
    url = data['url']  # Extract the URL from JSON data

    # Transform URL using vectorizer
    url_tfidf = vectorizer.transform([url])

    # Make prediction using the loaded model
    prediction = model.predict(url_tfidf)

    # Convert prediction to standard Python types (int, str, etc.)
    prediction = prediction.item()  # Convert numpy int64 to Python int

    # Return prediction as JSON response
    return jsonify({'prediction': prediction})

if __name__ == '__main__':
    app.run(debug=True)
