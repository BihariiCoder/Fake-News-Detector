#  Fake News Detection System

##  Project Overview

The Fake News Detection System is a Machine Learning-based web application that classifies news headlines as **Real** or **Fake**.
It uses Natural Language Processing (NLP) techniques and a trained model to analyze textual patterns in news data.

---

##  Objective

The main objective of this project is to:

* Detect fake news using Machine Learning
* Build a simple and interactive web interface
* Understand text classification using NLP techniques

---

##  Technologies Used

* Python
* Scikit-learn
* Pandas & NumPy
* Streamlit
* Natural Language Processing (NLP)

---

##  Project Structure

```
Fake-News-Detector/
│── app.py                # Streamlit UI
│── model.py              # Model training script
│── predict.py            # Prediction logic
│── dataset.csv           # Dataset (Kaggle)
│── model.pkl             # Saved trained model
│── vectorizer.pkl        # TF-IDF vectorizer
│── requirements.txt      # Dependencies
│── .streamlit/config.toml# Streamlit config
```

---

##  How It Works

1. Dataset is loaded and preprocessed
2. Text cleaning is applied (lowercase, remove symbols)
3. TF-IDF converts text into numerical features
4. Model is trained using Naive Bayes
5. Dataset is balanced to avoid bias
6. Model predicts whether news is Fake or Real

---

##  Installation & Setup

### 1. Clone Repository / Download Files

```bash
git clone <your-repo-link>
cd Fake-News-Detector
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Train Model

```bash
python model.py
```

### 4. Run Application

```bash
streamlit run app.py
```

---

##  Example Inputs

###  Fake News

* "Aliens landed in India yesterday"
* "Magic pill cures all diseases instantly"

### Real News

* "Government announces new policy"
* "Stock market reaches new high"

---

##  Key Features

* Real-time news classification
* Simple and interactive UI
* Balanced dataset handling
* High accuracy using Naive Bayes
* Fast prediction using saved model

---

##  Challenges Faced

* Dataset imbalance (more real news than fake)
* Model bias towards majority class
* Fixed using data balancing and proper model selection

---

##  Future Improvements

* Use Deep Learning models (LSTM, BERT)
* Analyze full news articles instead of headlines
* Add confidence score for predictions
* Deploy the project online

---

##  Author

**Ashish Ranjan**
B.Tech CSE Student

---

## 📌 Conclusion

This project demonstrates how Machine Learning and NLP can be used to detect fake news effectively. It highlights the importance of data preprocessing, model selection, and handling dataset imbalance.

---
