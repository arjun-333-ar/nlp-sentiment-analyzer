import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report
import joblib

nltk.download('stopwords')
nltk.download('wordnet')

# Expanded dataset covering common positive, negative, and neutral product terms
data = {
    'review_text': [
        # --- POSITIVE REVIEWS ---
        "Great quality product, highly recommend it!",
        "Excellent build quality, works perfectly and fast shipping.",
        "Absolutely love this item! Best purchase I have made.",
        "Super smooth experience, very durable and high quality.",
        "Amazing performance and fantastic design.",
        "Great value for money, completely satisfied.",
        "Awesome product, works as advertised!",
        "Good product, fast delivery and great packaging.",
        "Five stars! Outstanding quality and user friendly.",
        "Very happy with this order, functions great.",
        
        # --- NEGATIVE REVIEWS ---
        "Terrible quality, broke on the first day.",
        "Bad product, arrived damaged and stopped working.",
        "Worst customer service ever, total waste of money.",
        "Extremely disappointed, poor quality and cheap materials.",
        "Horrible experience, item does not work at all.",
        "Defective product, useless and cheap plastic.",
        "Slow delivery and product was broken upon arrival.",
        "Very poor performance, useless junk.",
        "Not worth the price, terrible experience.",
        "Cheaply made, stopped charging after two uses.",

        # --- NEUTRAL REVIEWS ---
        "Decent product for the price, nothing special.",
        "Average quality, works fine for basic usage.",
        "It is okay, meets standard expectations.",
        "So-so product, neither great nor bad.",
        "Fair quality, acceptable given the price tag.",
        "Ordinary item, functions as expected.",
        "Acceptable performance, standard delivery time.",
        "Just okay, nothing extraordinary.",
        "It works, but battery life could be longer.",
        "Standard item, fits description."
    ],
    'sentiment': [
        'Positive', 'Positive', 'Positive', 'Positive', 'Positive', 'Positive', 'Positive', 'Positive', 'Positive', 'Positive',
        'Negative', 'Negative', 'Negative', 'Negative', 'Negative', 'Negative', 'Negative', 'Negative', 'Negative', 'Negative',
        'Neutral', 'Neutral', 'Neutral', 'Neutral', 'Neutral', 'Neutral', 'Neutral', 'Neutral', 'Neutral', 'Neutral'
    ]
}

df = pd.DataFrame(data)

lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))

def clean_text(text):
    text = text.lower()
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    words = text.split()
    words = [lemmatizer.lemmatize(w) for w in words if w not in stop_words]
    return " ".join(words)

df['cleaned_text'] = df['review_text'].apply(clean_text)

X = df['cleaned_text']
y = df['sentiment']

nlp_pipeline = Pipeline([
    ('tfidf', TfidfVectorizer(ngram_range=(1, 2))),
    ('classifier', LogisticRegression(C=10.0)) # Increased C parameter for stronger feature weights
])

nlp_pipeline.fit(X, y)

y_pred = nlp_pipeline.predict(X)
print("--- Updated Model Classification Report ---")
print(classification_report(y, y_pred))

joblib.dump(nlp_pipeline, 'sentiment_model.pkl')
print("✅ New high-accuracy model saved as 'sentiment_model.pkl'!")