from transformers import AutoModelForSequenceClassification, AutoTokenizer
import torch

# Kelas untuk Klasifikasi Question
class MedQuestionClassifier(object):
  def __init__(self, path, device = None) -> None:
    self.path = path
    self.device = device if device != None else torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    self.load_model()
    self.label_mapper = {
      0: "non-medical",
      1: "medical"
    }

  # load model yang sudah disimpan dari hasil eksperimen
  def load_model(self):
    self.model = AutoModelForSequenceClassification.from_pretrained(self.path)
    self.tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
    self.model.to(self.device)

  # prediksi text input dengan model yang telah dibuat
  def predict(self, text):
    with torch.no_grad():
      inputs = self.tokenizer(text, return_tensors="pt")
      inputs.to(self.device)
      outputs = self.model(**inputs)
      logits = outputs[0]
      return self.label_mapper[logits.argmax(-1).item()]
