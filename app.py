import streamlit as st
import pandas as pd
import joblib
import os
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

#check if model files exist
if not os.path.exists("model/diagnosis_model.pkl") or not os.path.exists("model/tfid_vectorizer.pkl"):

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
        st.success(f"🤖 Based on your input, you may have: **{prediction}**")
        st.warning("⚠️ This is an AI prediction. Do not trust it 100%. Consult a doctor for accurate diagnosis.")
else:
    st.error("Please enter symptoms.")