import os
import pickle
import pandas as pd
from flask import Flask, request, jsonify
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB

app = Flask(__name__)
MODEL_PATH = "model.pkl"

def train_model(data_file):
    data = pd.read_csv(data_file)
    if 'text' not in data.columns or 'label' not in data.columns:
        return "CSV must have 'text' and 'label' columns.", 400
    
    texts = data['text']
    labels = data['label']
    
    vectorizer = CountVectorizer()
    X = vectorizer.fit_transform(texts)
    
    model = MultinomialNB()
    model.fit(X, labels)
    
    with open(MODEL_PATH, "wb") as f:
        pickle.dump((vectorizer, model), f)
    
    return "Model trained and saved successfully.", 200

@app.route("/train", methods=["POST"])
def train():
    file_path = "classification.csv"  # Use the existing CSV file
    
    if not os.path.exists(file_path):
        return jsonify({"error": "File classification.csv not found"}), 400

    msg, status = train_model(file_path)
    return jsonify({"message": msg}), status



@app.route("/predict", methods=["GET"])
def predict():
    try:
        data = request.get_json(force=True)  # Ensures JSON is parsed properly
    except Exception as e:
        return jsonify({"error": f"Invalid JSON input: {str(e)}"}), 400
    
    if not data or "text" not in data:
        return jsonify({"error": "No input text provided"}), 400

    text = data["text"]

    if not os.path.exists(MODEL_PATH):
        return jsonify({"error": "Model not trained yet"}), 400

    with open(MODEL_PATH, "rb") as f:
        vectorizer, model = pickle.load(f)

    X = vectorizer.transform([text])
    prediction = model.predict(X)[0]

    return jsonify({"prediction": prediction})

if __name__ == "__main__":
    app.run(debug=True)
