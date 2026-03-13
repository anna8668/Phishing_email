# train_model.py
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
import joblib

# ---------- 1. Load advanced cleaned dataset ----------
df = pd.read_csv("phishing_email_clean_advanced.csv")

# ---------- 1.1 Fill NaNs in clean_text ----------
# Ensure every document is a string (TF-IDF cannot handle NaN)
df['clean_text'] = df.get('clean_text', df.get('text_combined', "")).fillna("").astype(str)

# ---------- 2. Split features & labels ----------
X = df['clean_text']
y = df['label']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# ---------- 3. TF-IDF vectorization ----------
vectorizer = TfidfVectorizer(stop_words='english', max_features=10000, ngram_range=(1,2))
X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

# ---------- 4. Train Logistic Regression ----------
model = LogisticRegression(max_iter=1000, solver='saga')
model.fit(X_train_tfidf, y_train)

# ---------- 5. Evaluate ----------
y_pred = model.predict(X_test_tfidf)
print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))
print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))

# ---------- 6. Save model & vectorizer ----------
joblib.dump(model, "phish_model_advanced.joblib")
joblib.dump(vectorizer, "tfidf_vectorizer_advanced.joblib")
print("\n✅ Model and vectorizer saved as phish_model_advanced.joblib & tfidf_vectorizer_advanced.joblib")

# ---------- 7. Quick test function ----------
def predict_email(text):
    text = "" if text is None else str(text)
    vec = vectorizer.transform([text])
    pred = model.predict(vec)[0]
    prob = model.predict_proba(vec).max()
    return ("PHISHING" if pred==1 else "SAFE", prob)

# Test
if __name__ == "__main__":
    sample_email = "Please click http://fake-login.com to verify your account"
    label, prob = predict_email(sample_email)
    print("\nSample prediction:", label, f"(prob={prob:.3f})")
