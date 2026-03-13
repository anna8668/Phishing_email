import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# nltk.download('stopwords')
# nltk.download('wordnet')

stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

df = pd.read_csv("phishing_email.csv")

def clean_text_advanced(text):
    text = str(text).lower()
    text = re.sub(r'http\S+', ' url ', text)
    text = re.sub(r'www\.\S+', ' url ', text)
    text = re.sub(r'\S+@\S+', ' emailaddr ', text)
    text = re.sub(r'\d+', ' number ', text)
    text = re.sub(r'[^\w\s]', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    tokens = [lemmatizer.lemmatize(word) for word in text.split() if word not in stop_words]
    return " ".join(tokens)

df['clean_text'] = df['text_combined'].apply(clean_text_advanced)
df['clean_text'] = df['clean_text'].fillna("")

df.to_csv("phishing_email_clean_advanced.csv", index=False)
print("✅ Cleaned CSV saved as phishing_email_clean_advanced.csv")
print("\nSample:\n", df[['text_combined','clean_text','label']].head(3).to_string(index=False))
