import joblib

# Load saved model and vectorizer
model = joblib.load("phish_model.joblib")
vectorizer = joblib.load("tfidf_vectorizer.joblib")

def predict_email(text):
    vec = vectorizer.transform([text])
    pred = model.predict(vec)[0]
    prob = model.predict_proba(vec).max()
    return ("PHISHING" if pred == 1 else "SAFE", prob)

print("🚀 Phishing Email Detector")
print("Type 'exit' to quit\n")

while True:
    email_text = input("Enter email content: ")
    if email_text.lower() == "exit":
        break
    label, prob = predict_email(email_text)
    print(f"Prediction: {label} (Confidence: {prob:.3f})\n")
