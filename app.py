import streamlit as st
import pandas as pd
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

#Load Model and Vectorizer
model = joblib.load("model/diagnosis_model.pkl")
vectorizer = joblib.load("model/tfidf_vectorizer.pkl")

st.title("🩺 AI Medical Diagnosis")
st.write("Enter your symptoms and get a possible disease prediction.(Disclaimer: Do not trust 100%)")

#Input
symptom_input = st.text_area("📝 Describe your symptoms (e.g., fever, cough, body pain)")

if st.button("🔍 Diagnose"):
    if symptom_input.strip():
        X_input = vectorizer.transform([symptom_input])
        prediction = model.predict(X_input)[0]