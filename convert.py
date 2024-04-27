import torch
from transformers import DistilBertForSequenceClassification, DistilBertTokenizer
import os

MODEL_PATH = '/Users/vikranthbakkashetty/Desktop/major_project/tweet-classifier/src/model.py'


if os.path.exists(MODEL_PATH):
    tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
    model = DistilBertForSequenceClassification.from_pretrained('distilbert-base-uncased', num_labels=2)

    try:
        model.load_state_dict(torch.load(MODEL_PATH, map_location=torch.device('cpu')))
        print("Model loaded successfully.")
    except Exception as e:
        print("Error loading model weights:", e)
else:
    print(f"Error: Model file '{MODEL_PATH}' not found.")
