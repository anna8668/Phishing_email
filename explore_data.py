import pandas as pd

# ---------- 1. Load the dataset ----------
try:
    df = pd.read_csv("phishing_email.csv")
    print("✅ Dataset loaded successfully!\n")
except FileNotFoundError:
    print("❌ Error: phishing_email.csv not found in this folder.")
    exit()

# ---------- 2. Show basic info ----------
print("📊 Dataset Shape:", df.shape)
print("\n🧾 Columns:", df.columns.tolist())

# ---------- 3. Show sample rows ----------
print("\n🔍 Sample Rows:\n", df.head(5).to_string(index=False))

# ---------- 4. Missing values ----------
print("\n🚨 Missing Values per Column:\n", df.isnull().sum())

# ---------- 5. Quick look at label column ----------
for c in df.columns:
    if 'label' in c.lower() or 'phish' in c.lower():
        print(f"\n🎯 Unique values in '{c}':", df[c].unique())
