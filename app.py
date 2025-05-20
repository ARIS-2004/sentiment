import streamlit as st
import joblib
import re
import nltk
from nltk.corpus import stopwords

nltk.download('stopwords')

def clean_text(text):
    text = text.lower()
    text = re.sub(r'http\S+', '', text)
    text = re.sub(r'[^a-z\s]', '', text)
    tokens = text.split()
    stops = set(stopwords.words('english'))
    tokens = [word for word in tokens if word not in stops]
    return ' '.join(tokens)

# Load model and vectorizer
model = joblib.load('sentiment_model.joblib')
vectorizer = joblib.load('tfidf_vectorizer.joblib')

st.set_page_config(page_title="Tweet Sentiment Analysis", layout="centered")

st.title("📝 Tweet Sentiment Analyzer")
st.markdown(
    """
    Enter any tweet or sentence below, and the app will predict if it's **Positive**, **Negative**, or **Neutral**.
    """
)

user_input = st.text_area("Enter Text", height=120)

if st.button("Analyze Sentiment"):
    if not user_input.strip():
        st.warning("Please enter some text to analyze.")
    else:
        cleaned = clean_text(user_input)
        vectorized = vectorizer.transform([cleaned])
        prediction = model.predict(vectorized)[0]

        colors = {
            "positive": "green",
            "negative": "red",
            "neutral": "gray"
        }
        color = colors.get(prediction.lower(), "black")

        st.markdown(f"<h3 style='color:{color};'>Predicted Sentiment: {prediction.capitalize()}</h3>", unsafe_allow_html=True)
