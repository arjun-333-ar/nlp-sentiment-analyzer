# 💬 E-Commerce Product Review Sentiment Analyzer

An end-to-end **Natural Language Processing (NLP)** application that classifies customer feedback into **Positive**, **Negative**, or **Neutral** sentiment in real time with model confidence breakdowns.

![Python](https://img.shields.io/badge/Python-3.9+-3776AB?style=flat&logo=python&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-App-FF4B4B?style=flat&logo=streamlit&logoColor=white)
![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-ML-F7931E?style=flat&logo=scikit-learn&logoColor=white)

---

## 📌 Business Problem
Online retail platforms process thousands of customer reviews every day. Manually sifting through feedback to identify unhappy customers is inefficient. 

This tool automates feedback triaging by detecting negative reviews instantly, allowing customer support teams to prioritize high-risk issues and mitigate churn.

---

## 🛠️ Tech Stack & Methodology
- **Language:** Python
- **User Interface:** Streamlit
- **Machine Learning & NLP:** Scikit-Learn, NLTK, Joblib, Pandas
- **Text Preprocessing Pipeline:**
  - Tokenization & Lowercasing
  - Punctuation & Special Character Removal
  - Stop-Word Filtering (NLTK)
  - Word Lemmatization (`WordNetLemmatizer`)
- **Feature Extraction:** TF-IDF Vectorization (`ngram_range=(1, 2)`)
- **Classifier:** Logistic Regression Pipeline

---

## 📂 Project Architecture

```text
nlp-sentiment-analyzer/
├── app.py                 # Streamlit Web Application interface
├── train_nlp.py           # Model training & preprocessing pipeline script
├── sentiment_model.pkl    # Serialized ML pipeline model
├── requirements.txt       # Python dependencies list
└── README.md              # Project documentation
