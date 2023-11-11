from med_question_classifier import MedQuestionClassifier
from sentiment import SentimentClassifier

def main():
  med_question_clf = MedQuestionClassifier("./medical-bert-classifier")
  sentiment_clf = SentimentClassifier("./standard-bert-sentiment")
  
  while(True):
    text = input("Enter a sentence: ")
    res = med_question_clf.predict(text)
    print("This sentence is", res)
    if res == "medical":
      print(sentiment_clf.predict(text))
    else:
      return
    print("")

if __name__ == "__main__":
  main()