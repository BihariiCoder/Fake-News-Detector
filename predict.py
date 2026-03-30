import pickle
import re

model = pickle.load(open("model.pkl", "rb"))
vectorizer = pickle.load(open("vectorizer.pkl", "rb"))

def clean_text(text):
    text = str(text).lower()
    text = re.sub(r'[^a-zA-Z]', ' ', text)
    return text

def predict_news(news):
    news = clean_text(news)
    vector = vectorizer.transform([news])
    return model.predict(vector)[0]