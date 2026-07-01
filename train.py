"""
Fake/Genuine Review Detection — TF-IDF + XGBoost
=================================================
Runs entirely on CPU. No GPU needed.

Install deps:
    !pip install -q xgboost scikit-learn pandas numpy joblib

Dataset: fake_reviews_dataset.csv
  columns: category, rating, label (CG=fake, OR=genuine), text_
"""

import pandas as pd
import numpy as np
import joblib
import os

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import (
    classification_report, confusion_matrix,
    accuracy_score, roc_auc_score
)
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBClassifier

# ----------------------------
# Config
# ----------------------------
CSV_PATH   = "fake_reviews_dataset.csv"
OUTPUT_DIR = "./fake_review_detector"
SEED       = 42

os.makedirs(OUTPUT_DIR, exist_ok=True)

# ----------------------------
# 1. Load data
# ----------------------------
df = pd.read_csv(CSV_PATH).dropna(subset=["text_", "label"]).reset_index(drop=True)

# Encode label: OR=0 (genuine), CG=1 (fake)
df["label_id"] = (df["label"] == "CG").astype(int)

# Optional: combine rating as extra feature signal
df["text_with_rating"] = df["rating"].astype(str) + " stars. " + df["text_"]

X = df["text_with_rating"]
y = df["label_id"]

# ----------------------------
# 2. Train / Val / Test split  80 / 10 / 10
# ----------------------------
X_train, X_temp, y_train, y_temp = train_test_split(
    X, y, test_size=0.2, random_state=SEED, stratify=y
)
X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp, test_size=0.5, random_state=SEED, stratify=y_temp
)
print(f"Train: {len(X_train)}  Val: {len(X_val)}  Test: {len(X_test)}")

# ----------------------------
# 3. TF-IDF vectorizer
# ----------------------------
print("Fitting TF-IDF...")
tfidf = TfidfVectorizer(
    max_features=50_000,    # top 50k terms
    ngram_range=(1, 2),     # unigrams + bigrams
    sublinear_tf=True,      # log-scale TF
    min_df=2,
    strip_accents="unicode",
    analyzer="word",
)
X_train_tfidf = tfidf.fit_transform(X_train)
X_val_tfidf   = tfidf.transform(X_val)
X_test_tfidf  = tfidf.transform(X_test)
print(f"Vocabulary size: {len(tfidf.vocabulary_)}")

# ----------------------------
# 4. XGBoost
# ----------------------------
print("Training XGBoost...")
model = XGBClassifier(
    n_estimators=300,
    max_depth=6,
    learning_rate=0.1,
    subsample=0.8,
    colsample_bytree=0.8,
    use_label_encoder=False,
    eval_metric="logloss",
    random_state=SEED,
    n_jobs=-1,              # use all CPU cores
    tree_method="hist",     # fastest CPU method
)

model.fit(
    X_train_tfidf, y_train,
    eval_set=[(X_val_tfidf, y_val)],
    verbose=50,
)

# ----------------------------
# 5. Evaluate on test set
# ----------------------------
print("\n=== Test Set Results ===")
y_pred      = model.predict(X_test_tfidf)
y_pred_prob = model.predict_proba(X_test_tfidf)[:, 1]

print(f"Accuracy : {accuracy_score(y_test, y_pred):.4f}")
print(f"ROC-AUC  : {roc_auc_score(y_test, y_pred_prob):.4f}")
print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=["genuine (OR)", "fake (CG)"]))
print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))

# ----------------------------
# 6. Save model + vectorizer
# ----------------------------
joblib.dump(tfidf,  os.path.join(OUTPUT_DIR, "tfidf.pkl"))
joblib.dump(model,  os.path.join(OUTPUT_DIR, "xgb_model.pkl"))
print(f"\nSaved to {OUTPUT_DIR}/tfidf.pkl and xgb_model.pkl")
