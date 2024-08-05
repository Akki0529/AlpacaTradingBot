from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
from typing import Tuple

# Determine device
device = "cuda:0" if torch.cuda.is_available() else "cpu"

# Load FinBERT model and tokenizer
tokenizer = AutoTokenizer.from_pretrained("ProsusAI/finbert")
model = AutoModelForSequenceClassification.from_pretrained("ProsusAI/finbert").to(device)
labels = ["positive", "negative", "neutral"]

def estimate_sentiment(news: str) -> Tuple[float, str]:
    if news:
        tokens = tokenizer(news, return_tensors="pt", padding=True).to(device)
        result = model(tokens["input_ids"], attention_mask=tokens["attention_mask"])["logits"]
        result = torch.nn.functional.softmax(torch.sum(result, 0), dim=-1)
        probability = result[torch.argmax(result)].item()
        sentiment = labels[torch.argmax(result)]
        return probability, sentiment
    else:
        return 0, labels[-1]

if __name__ == "__main__":
    news = ['markets responded negatively to the news!', 'traders were happy!']
    for article in news:
        probability, sentiment = estimate_sentiment(article)
        print(f"Sentiment: {sentiment}, Probability: {probability:.4f}")

    print("CUDA available:", torch.cuda.is_available())
