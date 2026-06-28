"""
Train Model - Deteksi Spam SMS
Dataset: SMS Spam Collection (5572 SMS)
Menggunakan TF-IDF + Logistic Regression
"""

import pandas as pd
import numpy as np
import re
import string
import joblib
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix

# ─── 1. Load Dataset ──────────────────────────────────────────────────────────
url = "https://raw.githubusercontent.com/justmarkham/pycon-2016-tutorial/master/data/sms.tsv"
df = pd.read_csv(url, sep='\t', header=None, names=['label', 'teks'])
print(f"Dataset: {df.shape[0]} SMS")
print(f"Distribusi label:\n{df['label'].value_counts()}")

# ─── 2. Preprocessing ─────────────────────────────────────────────────────────
def preprocess(text):
    text = text.lower()                                                        # Case folding
    text = re.sub(r'http\S+|www\S+', '', text)                                # Hapus URL
    text = re.sub(r'[0-9]+', '', text)                                        # Hapus angka
    text = text.translate(str.maketrans('', '', string.punctuation))           # Hapus tanda baca
    text = ' '.join(text.split())                                              # Hapus spasi berlebih
    return text

df['teks_bersih'] = df['teks'].apply(preprocess)
print("\nContoh preprocessing:")
print(df[['teks', 'teks_bersih']].head(3).to_string())

# ─── 3. Split Data ────────────────────────────────────────────────────────────
X_train, X_test, y_train, y_test = train_test_split(
    df['teks_bersih'], df['label'],
    test_size=0.2, random_state=42, stratify=df['label']
)
print(f"\nData latih: {len(X_train)}, Data uji: {len(X_test)}")

# ─── 4. Ekstraksi Fitur TF-IDF ───────────────────────────────────────────────
vectorizer = TfidfVectorizer(ngram_range=(1, 2), max_features=5000)
X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf  = vectorizer.transform(X_test)
print(f"Ukuran vocabulary: {len(vectorizer.vocabulary_)}")

# ─── 5. Training & Perbandingan Model ────────────────────────────────────────
# Naive Bayes
nb = MultinomialNB()
nb.fit(X_train_tfidf, y_train)
nb_pred = nb.predict(X_test_tfidf)
print(f"\nNaive Bayes         → Accuracy: {accuracy_score(y_test, nb_pred):.4f} | F1: {f1_score(y_test, nb_pred, pos_label='spam'):.4f}")

# Logistic Regression
lr = LogisticRegression(random_state=42, max_iter=1000)
lr.fit(X_train_tfidf, y_train)
lr_pred = lr.predict(X_test_tfidf)
print(f"Logistic Regression → Accuracy: {accuracy_score(y_test, lr_pred):.4f} | F1: {f1_score(y_test, lr_pred, pos_label='spam'):.4f}")

# ─── 6. Evaluasi Model Final ─────────────────────────────────────────────────
print("\n=== Evaluasi Model Final (Logistic Regression) ===")
print(classification_report(y_test, lr_pred, target_names=['Ham', 'Spam']))
print("Confusion Matrix:")
print(confusion_matrix(y_test, lr_pred))

# ─── 7. Simpan Model & Vectorizer ────────────────────────────────────────────
joblib.dump(lr, 'model.pkl')
joblib.dump(vectorizer, 'vectorizer.pkl')
print("\nModel disimpan ke model.pkl")
print("Vectorizer disimpan ke vectorizer.pkl")
