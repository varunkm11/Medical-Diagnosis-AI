# 🩺 AI Medical Diagnosis App

This AI-powered medical diagnosis app takes user-input symptoms and predicts the most likely disease using a machine learning model trained on medical symptom data.

> ⚠️ This tool is for educational/demo purposes only. For real medical concerns, always consult a professional doctor.

---

## 🚀 Demo

Click here for --[Live demo](https://medical-diagnosis-ai-sev4d9zqemgahdwrxjesky.streamlit.app/) 

---

## 📌 Features

- ✅ Symptom-based disease prediction
- ✅ Trained using TF-IDF + Logistic Regression
- ✅ Confidence score shown for prediction
- ✅ Clean and user-friendly Streamlit UI
- ✅ Easy retraining with your own CSV data

---

## 🗂️ Project Structure
```bash
AI-Medical-Diagnosis-App/
│
├── app.py                    # Streamlit app for AI-based disease diagnosis
├── model_training.py         # Script to train and save the ML model and vectorizer
├── README.md                 # Project documentation
├── requirements.txt          # List of Python dependencies
│
├── data/
│   └── medical_data.csv      # Dataset containing symptoms and disease labels
│
└── model/
    ├── diagnosis_model.pkl       # Trained Logistic Regression model
    └── tfidf_vectorizer.pkl      # TF-IDF vectorizer used for transforming input
```

---

## 🧠 How it Works

1. The app takes user input (symptoms).
2. It vectorizes the input using **TF-IDF**.
3. The input is passed to a **Logistic Regression** model trained on labeled symptoms.
4. The app displays the predicted disease and confidence score.

---

## 🔧 Setup Instructions

### 1. Clone the repository

```bash
git clone https://github.com/your-username/AI-Medical-Diagnosis-App.git
cd AI-Medical-Diagnosis-App
```
### 2. Install dependencies
```bash
pip install -r requirements.txt
```
### 3. Train the model
```bash
python model_training.py
```
### 4. streamlit run app.py
```bash
streamlit run app.py
```
## 📌 Sample Input
```bash
fever, dry cough, loss of taste or smell
```
## ✅ Output
```bash
🤖 Based on your input, you may have: COVID-19
🧠 AI Confidence: 91.23%
⚠️ This is an AI prediction. Do not trust it 100%. Consult a doctor for accurate diagnosis.
```
## 🙌 Contributions
Feel free to contribute to improve the model, UI, or data. Pull requests are welcome!
## 🔗 Author
👤 [varunkm11](https://github.com/varunkm11)


