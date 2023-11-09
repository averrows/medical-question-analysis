from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
from sentiment import SentimentClassifier

app = Flask(__name__,static_folder='static')
CORS(app, origins=["*"])
MODELS = {
  "sentiment": SentimentClassifier("./standard-bert-sentiment"),
}

@app.route("/")
def main():
  return render_template('index.html')

@app.route("/nlp/<string:task>/predict", methods=["GET"])
def predict(task):
  if task not in MODELS:
    return "Task not found", 404
  text = request.args.get("text", default="", type=str)
  if text == "":
    return "Fill in the text query to predict", 404
  model = MODELS[task]

  response = {
    "task": task,
    "text": text,
    "prediction": model.predict(text)
  }

  return jsonify(response)