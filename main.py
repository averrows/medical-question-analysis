from sentiment import SentimentClassifier
def main():
  sentiment_clf = SentimentClassifier("averrous/standard-bert-sentiment-classifier")
  
  while(True):
    text = input("Enter a sentence: ")
    print(sentiment_clf.predict(text))
    print("")


if __name__ == "__main__":
  main()