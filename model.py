import torch
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification, TrainingArguments, Trainer
from torch.utils.data import DataLoader, Dataset
from sklearn.model_selection import train_test_split
import pandas as pd

# Replace with your dataset path
DATASET_PATH = '/Users/vikranthbakkashetty/Desktop/major_project/tweet-classifier/src/train.csv'
MODEL_PATH = '/Users/vikranthbakkashetty/Desktop/major_project/tweet-classifier/src/model'  # Change as needed
MAX_LENGTH = 128  # Adjust based on your requirements

class TweetDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = str(self.texts.iloc[idx])  # Use iloc to access rows by integer index
        label = int(self.labels.iloc[idx])
        encoding = self.tokenizer(text, truncation=True, padding='max_length', max_length=self.max_length, return_tensors='pt')

        return {
            'text': text,
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'label': torch.tensor(label, dtype=torch.long)
        }

class FineTunedDistilBERTClassifier:
    def __init__(self, model_path=MODEL_PATH, max_length=MAX_LENGTH):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model_path = model_path
        self.max_length = max_length
        self.tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
        self.model = self.load_or_initialize_model()
        self.model.eval()

    def load_or_initialize_model(self):
        model = DistilBertForSequenceClassification.from_pretrained('distilbert-base-uncased', num_labels=2)

        try:
            model.load_state_dict(torch.load(self.model_path, map_location=self.device))
            print("Loaded fine-tuned DistilBERT model.")
        except FileNotFoundError:
            print(f"Model file '{self.model_path}' not found. Initializing a new model.")
            model.save_pretrained(self.model_path)  # Save the initialized model weights
        except Exception as e:
            print(f"Error loading the model: {e}. Initializing a new model.")

        return model.to(self.device)

    def predict(self, tweet_text):
        inputs = self.tokenizer(tweet_text, truncation=True, padding='max_length', max_length=self.max_length, return_tensors='pt')
        inputs = {key: inputs[key].to(self.device) for key in inputs}

        with torch.no_grad():
            outputs = self.model(**inputs)
            logits = outputs.logits

        probabilities = torch.nn.functional.softmax(logits, dim=1)
        prediction = torch.argmax(probabilities, dim=1).item()

        return prediction, probabilities[0][1].item()

if __name__ == "__main__":
    # Load your dataset
    data = pd.read_csv(DATASET_PATH)

    # Reset index to avoid index-related issues
    data = data.reset_index(drop=True)

    # Handle missing values if any
    data = data.dropna(subset=['text', 'target'])

    # Assuming your dataset has 'text' column for tweets and 'target' column for labels
    train_texts, test_texts, train_labels, test_labels = train_test_split(data['text'], data['target'], test_size=0.2, random_state=42)

    # Initialize tokenizer
    tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')

    # Initialize and fine-tune the model
    model = FineTunedDistilBERTClassifier()
    train_dataset = TweetDataset(train_texts, train_labels, tokenizer, MAX_LENGTH)
    training_args = TrainingArguments(
        output_dir='./results',
        num_train_epochs=3,
        per_device_train_batch_size=4,  # Adjust based on your resources
        save_steps=500,
        save_total_limit=2,
    )
    trainer = Trainer(
        model=model.model,
        args=training_args,
        train_dataset=train_dataset,
    )
    trainer.train()

    # Save the fine-tuned model
    model.model.save_pretrained(MODEL_PATH)
