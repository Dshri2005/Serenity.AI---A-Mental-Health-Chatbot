from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import mysql.connector
from datetime import datetime

# Model configuration
MODEL_CONFIGS = {
    "distilbert": {
        "model_name": "distilbert-base-uncased-finetuned-sst-2-english",
        "sentiment_map": {0: "Negative", 1: "Positive"}
    },
}

# Load model and tokenizer
model_key = "distilbert"
config = MODEL_CONFIGS[model_key]
tokenizer = AutoTokenizer.from_pretrained(config["model_name"])
model = AutoModelForSequenceClassification.from_pretrained(config["model_name"])

# Sentiment Analysis Function
def sentiment_analysis(msg):
    inputs = tokenizer(msg, return_tensors="pt", truncation=True, padding=True, max_length=512)
    with torch.no_grad():
        outputs = model(**inputs)
    probs = torch.nn.functional.softmax(outputs.logits, dim=-1)
    sentiment_map = config["sentiment_map"]
    return [sentiment_map[p] for p in torch.argmax(probs, dim=-1).tolist()]

# Store Sentiment to MySQL (using user_id)
def store_sentiment_to_mysql(sentiment, user_id):
    try:
        conn = mysql.connector.connect(
            host="localhost",
            user="root",
            password="Dshr!2022",
            database="mh_chatbot"
        )
        cursor = conn.cursor()
        query = "INSERT INTO sentiments (sentiment, user_id) VALUES (%s, %s)"
        cursor.execute(query, (sentiment, user_id))
        conn.commit()
        cursor.close()
        conn.close()
    except Exception as e:
        print(f"❌ Error storing sentiment: {e}")


#\connect root@localhost
