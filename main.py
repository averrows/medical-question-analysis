from med_question_classifier import MedQuestionClassifier
from sentiment import SentimentClassifier
from similarity import SimilarityClassifier

# Main program
def main():
  # Load kelas tiap model
  med_question_clf = MedQuestionClassifier("./medical-bert-classifier")
  sentiment_clf = SentimentClassifier("./standard-bert-sentiment")
  similarity_clf = SimilarityClassifier("./question-similarity-bert")
  
  while(True):
    print('='*40)
    print()
    # input pertanyaan pertama
    question_1 = input("Enter first question: ")
    # medical question classification
    res = med_question_clf.predict(question_1)
    print()
    print(">> This question is", res)
    if res == "medical": 
      print(">> The sentiment is", sentiment_clf.predict(question_1)) # analisis sentimen
      print()
      # input pertanyaan kedua
      question_2 = input("Enter second question: ")
      print()
      res = med_question_clf.predict(question_2)
      print(">> This question is", res)
      if res == "medical":
        # similarity checking kedua pertanyaan
        print(">> The sentiment is", sentiment_clf.predict(question_2)) # analisis sentimen
        print()
        print(">> These two question is", similarity_clf.predict(question_1, question_2))
      print()
    print()

if __name__ == "__main__":
  main()