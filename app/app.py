from flask import Flask, render_template, request
import pickle
import re

app = Flask(__name__)

# Load model
with open("../model/fake_news_model.pkl", "rb") as file:
    model = pickle.load(file)

# Load vectorizer
with open("../model/vectorizer.pkl", "rb") as file:
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