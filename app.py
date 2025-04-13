import streamlit as st
import pandas as pd
import joblib
import os
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

#check if model files exist
if not os.path.exists("model/diagnosis_model.pkl") or not os.path.exists("model/tfidf_vectorizer.pkl"):
    st.error("⚠️ Model files not found! Make sure you've trained and saved your model in the 'model' folder.")
    st.stop()
#Load Model and Vectorizer
model = joblib.load("model/diagnosis_model.pkl")
vectorizer = joblib.load("model/tfidf_vectorizer.pkl")

#streamlit App Title and description
st.markdown(
    """
    <h1 style='text-align:center;'>🩺 AI Medical Diagnosis App</h1>
    <p style='text-align:center;'>Enter your symptoms and get a possible disease prediction.<br>
    """,
    unsafe_allow_html=True
)

#Input
symptom_input = st.text_area("📝 Describe your symptoms (e.g., fever, cough, body pain)")

#diagnosis button
if st.button("🔍 Diagnose"):
    if symptom_input.strip():
        X_input = vectorizer.transform([symptom_input])
        prediction = model.predict(X_input)[0]
        try:
            probability = model.predict_proba(X_input).max()*100
            st.success(f"🤖 Based on your input, you may have: **{prediction}**")
            st.info(f"🧠 AI Confidence: {probability:.2f}%")
        except AttributeError:
            #if model doesn;t support predict_proba
                st.success(f"🤖 Based on your input, you may have: **{prediction}**")
        st.warning("⚠️ This is an AI prediction. Do not trust it 100%. Consult a doctor for accurate diagnosis.")
else:
    st.error("Please enter your symptoms  before clicking Diagnosis.")