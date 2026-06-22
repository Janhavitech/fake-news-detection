import streamlit as st
import joblib
import re

# Load saved model and vectorizer
model = joblib.load("models/fake_news_model.pkl")
vectorizer = joblib.load("models/tfidf_vectorizer.pkl")

# Basic text cleaning (same as training)
def clean_text(text):
    text = text.lower()
    text = re.sub(r'[^a-z\s]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

# App UI
st.set_page_config(page_title="Fake News Detection", layout="centered")

st.title("📰 Fake News Detection App")
st.write("Enter a news article text below to check whether it is **Fake** or **Real**.")

# Text input
user_input = st.text_area("Paste news article here:")

if st.button("Analyze"):
    if user_input.strip() == "":
        st.warning("Please enter some text.")
    else:
        cleaned = clean_text(user_input)
        vectorized = vectorizer.transform([cleaned])
        prediction = model.predict(vectorized)[0]

        if prediction == 0:
            st.error("🚨 This news is predicted as **FAKE**")
        else:
            st.success("✅ This news is predicted as **REAL**")
