from flask import Flask, request, jsonify
from flask_cors import CORS
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification
import torch

app = Flask(__name__)
CORS(app)

# Load the fine-tuned DistilBERT model
MODEL_PATH = '/Users/vikranthbakkashetty/Desktop/major_project/tweet-classifier/src/model'  # Update with your model path
tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
model = DistilBertForSequenceClassification.from_pretrained(MODEL_PATH)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)
model.eval()

def predict(tweet_text):
    try:
        inputs = tokenizer(tweet_text, truncation=True, padding=True, max_length=128, return_tensors='pt')
        inputs = {key: inputs[key].to(device) for key in inputs}

        with torch.no_grad():
            outputs = model(**inputs)
            logits = outputs.logits

        probabilities = torch.nn.functional.softmax(logits, dim=1)
        prediction = torch.argmax(probabilities, dim=1).item()

        return prediction, probabilities[0][1].item()
    except Exception as e:
        return None, str(e)

@app.route('/api/predict', methods=['POST'])
def predict_api():
    try:
        data = request.get_json()
        tweet_text = data['tweetText']

        svm_prediction, svm_probability = predict(tweet_text)

        if svm_prediction is not None:
            response_data = {
                'svm_prediction': svm_prediction,
                'svm_accuracy': svm_probability,
                'chartData': None  # Update with your chart data if available
            }
            return jsonify(response_data)
        else:
            return jsonify({'error': 'Prediction error'}), 500
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/previous-findings', methods=['GET'])
def get_previous_findings():
    # Implement logic to retrieve previous findings
    previous_findings = []  # Update with your logic to retrieve previous findings
    return jsonify(previous_findings)

if __name__ == '__main__':
    app.run(port=5000, debug=True)
