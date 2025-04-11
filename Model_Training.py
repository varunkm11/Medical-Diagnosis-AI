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
x_vec = vectorizer.fit_transform(x)

model = LogisticRegression()
model.fit(x_vec, y)

#save Model ANd vectorizer
os.makedirs("model, exist_ok=True")
joblib.dump(model, "model/diagnosis_model.pkl")
joblib.dump(vectorizer,"model/tfidf_vectorizer.pkl")