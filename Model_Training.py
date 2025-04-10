import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
import joblib
import os
#Load and preprocess
df = pd.read_csv("data/medical_data.csv")

x = df["symptoms"]
y = df["disease"]

vectorizer = TfidfVectorizer()
x_vec = vectorizer