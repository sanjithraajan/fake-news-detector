from flask import Flask, render_template, request
import pickle
import re

app = Flask(__name__)

# Load model
import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

model_path = os.path.join(BASE_DIR, "model", "fake_news_model.pkl")
vectorizer_path = os.path.join(BASE_DIR, "model", "vectorizer.pkl")

with open(model_path, "rb") as file:
    model = pickle.load(file)

with open(vectorizer_path, "rb") as file:
    vectorizer = pickle.load(file)

# Text cleaning function
def clean_text(text):
    text = text.lower()
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    return text

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():

    news = request.form["news"]

    clean_news = clean_text(news)

    vector_input = vectorizer.transform([clean_news])

    prediction = model.predict(vector_input)

    if prediction[0] == 0:
        result = "Fake News"
    else:
        result = "Real News"

    return render_template("index.html", prediction=result)

if __name__ == "__main__":
    app.run(debug=True)