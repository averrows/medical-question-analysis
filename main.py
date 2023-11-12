from med_question_classifier import MedQuestionClassifier
from sentiment import SentimentClassifier
from similarity import SimilarityClassifier

def main():
  med_question_clf = MedQuestionClassifier("./medical-bert-classifier")
  sentiment_clf = SentimentClassifier("./standard-bert-sentiment")
  similarity_clf = SimilarityClassifier("./question-similarity-bert")
  
  while(True):
    question_1 = input("Enter first question: ")
    res = med_question_clf.predict(question_1)
    print("This question is", res)
    if res == "medical":
      print("The sentiment is", sentiment_clf.predict(question_1))
      question_2 = input("Enter second question: ")
      res = med_question_clf.predict(question_2)
      print("This question is", res)
      if res == "medical":
        print("The sentiment is", sentiment_clf.predict(question_2))
        print("These two question is", similarity_clf.predict(question_1, question_2))
      print("")
    print("")

if __name__ == "__main__":
  main()