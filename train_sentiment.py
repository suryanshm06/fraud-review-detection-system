"""
Sentiment Analyzer — TF-IDF + XGBoost on Kindle Reviews
=========================================================
Predicts star rating (1-5) from review text, then buckets into:
    Negative (1-2) / Neutral (3) / Positive (4-5)

This model is combined with the fraud score in the web app to give:
    - Is this review fake? (fraud model)
    - What sentiment does it express? (this model)
    - Do the rating and sentiment actually match? (mismatch = extra fraud signal)

Install:
    !pip install -q xgboost scikit-learn pandas numpy joblib

Dataset: preprocessed_kindle_review_.csv
  columns: rating (1-5), reviewText, summary
"""

import os
import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from xgboost import XGBClassifier

# ----------------------------
# Config
# ----------------------------
CSV_PATH   = "preprocessed_kindle_review_.csv"
OUTPUT_DIR = "./fake_review_detector"
SEED       = 42

os.makedirs(OUTPUT_DIR, exist_ok=True)

# ----------------------------
# 1. Load & prepare
# ----------------------------
df = pd.read_csv(CSV_PATH).dropna(subset=["reviewText", "rating"]).reset_index(drop=True)

# Combine reviewText + summary for richer signal
df["summary"] = df["summary"].fillna("")
df["full_text"] = df["reviewText"] + " " + df["summary"]

# Bucket ratings into 3 sentiment classes
def to_sentiment(rating):
    if rating <= 2: return 0   # Negative
    if rating == 3: return 1   # Neutral
    return 2                   # Positive

df["sentiment"] = df["rating"].apply(to_sentiment)

SENT_LABELS = {0: "negative", 1: "neutral", 2: "positive"}
print("Sentiment distribution:")
print(df["sentiment"].map(SENT_LABELS).value_counts())

X = df["full_text"]
y = df["sentiment"]

# ----------------------------
# 2. Split  80 / 10 / 10
# ----------------------------
X_train, X_temp, y_train, y_temp = train_test_split(
    X, y, test_size=0.2, random_state=SEED, stratify=y
)
X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp, test_size=0.5, random_state=SEED, stratify=y_temp
)
print(f"\nTrain: {len(X_train)}  Val: {len(X_val)}  Test: {len(X_test)}")

# ----------------------------
# 3. TF-IDF
# ----------------------------
print("Fitting TF-IDF...")
tfidf_sent = TfidfVectorizer(
    max_features=40_000,
    ngram_range=(1, 2),
    sublinear_tf=True,
    min_df=2,
    strip_accents="unicode",
)
X_train_tfidf = tfidf_sent.fit_transform(X_train)
X_val_tfidf   = tfidf_sent.transform(X_val)
X_test_tfidf  = tfidf_sent.transform(X_test)

# ----------------------------
# 4. XGBoost (multi-class)
# ----------------------------
print("Training XGBoost sentiment model...")
sent_model = XGBClassifier(
    n_estimators=300,
    max_depth=6,
    learning_rate=0.1,
    subsample=0.8,
    colsample_bytree=0.8,
    objective="multi:softprob",
    num_class=3,
    eval_metric="mlogloss",
    random_state=SEED,
    n_jobs=-1,
    tree_method="hist",
)
sent_model.fit(
    X_train_tfidf, y_train,
    eval_set=[(X_val_tfidf, y_val)],
    verbose=50,
)

# ----------------------------
# 5. Evaluate
# ----------------------------
print("\n=== Test Set Results ===")
y_pred = sent_model.predict(X_test_tfidf)
print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=["negative", "neutral", "positive"]))
print("Confusion Matrix (neg / neu / pos):")
print(confusion_matrix(y_test, y_pred))

# ----------------------------
# 6. Save
# ----------------------------
joblib.dump(tfidf_sent,  os.path.join(OUTPUT_DIR, "tfidf_sentiment.pkl"))
joblib.dump(sent_model,  os.path.join(OUTPUT_DIR, "sentiment_model.pkl"))
print(f"\nSaved tfidf_sentiment.pkl and sentiment_model.pkl to {OUTPUT_DIR}/")
