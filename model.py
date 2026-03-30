import pandas as pd
import re
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB

# Load dataset
df = pd.read_csv("dataset.csv")

# Select columns
df = df[['title', 'real']]
df.columns = ['text', 'label']

# Remove null
df = df.dropna()

# 🔥 BALANCE DATASET
df_real = df[df['label'] == 1]
df_fake = df[df['label'] == 0]

df_real = df_real.sample(len(df_fake), random_state=42)

df = pd.concat([df_real, df_fake])

# Shuffle
df = df.sample(frac=1, random_state=42)

# Clean text
def clean_text(text):
    text = str(text).lower()
    text = re.sub(r'[^a-zA-Z]', ' ', text)
    return text

df['text'] = df['text'].apply(clean_text)

# Features & labels
X = df['text']
y = df['label']

# Vectorization
vectorizer = TfidfVectorizer(stop_words='english')
X = vectorizer.fit_transform(X)

# Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# 🔥 BETTER MODEL FOR TEXT
model = MultinomialNB()
model.fit(X_train, y_train)

# Save model
pickle.dump(model, open("model.pkl", "wb"))
pickle.dump(vectorizer, open("vectorizer.pkl", "wb"))

print("✅ Model trained successfully")