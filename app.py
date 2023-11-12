from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
from sentiment import SentimentClassifier
from similarity import SimilarityClassifier
from med_question_classifier import MedQuestionClassifier

app = Flask(__name__,static_folder='static')
CORS(app, origins=["*"])
MODELS = {
  "sentiment": SentimentClassifier("./standard-bert-sentiment"),
  "similarity": SimilarityClassifier("./question-similarity-bert"),
  "classification": MedQuestionClassifier("./medical-bert-classifier")
}

@app.route("/")
def main():
  return render_template('index.html')

@app.route("/nlp/<string:task>/predict", methods=["GET"])
def predict(task):
  if task not in MODELS:
    return "Task not found", 404
  
  model = MODELS[task]

  # similarity task
  if task == "similarity":
    # input field text kedua question
    text1 = request.args.get("text1", default="", type=str)
    text2 = request.args.get("text2", default="", type=str)

    # handling error
    if text1 == "" or text2 == "":
      return "Fill in the text1 and text2 query to predict", 404
    pred = model.predict(text1, text2)

  # classification and sentiment analysis task
  else:
    text = request.args.get("text", default="", type=str)
    if text == "":
      return "Fill in the text query to predict", 404
    pred = model.predict(text)

  response = {
    "task": task,
    "text": text if task != "similarity" else [text1, text2],
    "prediction": pred
  }

  return jsonify(response)