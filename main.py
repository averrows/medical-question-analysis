from sentiment import SentimentClassifier
def main():
  sentiment_clf = SentimentClassifier("./standard-bert-sentiment")
  
  while(True):
    text = input("Enter a sentence: ")
    print(sentiment_clf.predict(text))
    print("")


if __name__ == "__main__":
  main()