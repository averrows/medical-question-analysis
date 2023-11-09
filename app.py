from flask import Flask, request, jsonify
from sentiment import SentimentClassifier

app = Flask(__name__)


MODELS = {
  "sentiment": SentimentClassifier("./standard-bert-sentiment"),
}

@app.route("/")
def hello():
  return "Hello World!"

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