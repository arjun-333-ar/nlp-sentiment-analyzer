import streamlit as st
import joblib
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

nltk.download('stopwords')
nltk.download('wordnet')

lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))

def clean_text(text):
    text = text.lower()
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    words = text.split()
    words = [lemmatizer.lemmatize(w) for w in words if w not in stop_words]
    return " ".join(words)

# Load Model Pipeline
model = joblib.load('sentiment_model.pkl')

st.set_page_config(page_title="Product Review Sentiment Analyzer", page_icon="💬")

st.title("💬 E-Commerce Review Sentiment Analyzer")
st.write("Analyze customer feedback in real-time to detect sentiment and confidence.")

user_review = st.text_area("Enter Customer Review / Feedback:", height=120, placeholder="Type review here...")

if st.button("Analyze Sentiment", type="primary"):
    if user_review.strip() == "":
        st.warning("Please enter a review to analyze.")
    else:
        cleaned = clean_text(user_review)
        prediction = model.predict([cleaned])[0]
        probabilities = model.predict_proba([cleaned])[0]
        classes = model.classes_
        
        st.markdown("---")
        st.subheader("Analysis Results")
        
        if prediction == 'Positive':
            st.success("😊 Predicted Sentiment: **Positive**")
        elif prediction == 'Negative':
            st.error("😡 Predicted Sentiment: **Negative**")
            st.info("💡 Actionable Insight: Route directly to Customer Escalation Team.")
        else:
            st.info("😐 Predicted Sentiment: **Neutral**")
            
        st.write("**Model Confidence Breakdown:**")
        for cls, prob in zip(classes, probabilities):
            st.write(f"- **{cls}:** {prob * 100:.1f}%")