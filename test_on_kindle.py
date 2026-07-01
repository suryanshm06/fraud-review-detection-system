"""
Real-World Sanity Check — Run fraud model on Amazon Kindle reviews
==================================================================
No ground-truth fraud labels here, so this is an audit / inspection tool.

What it tells you:
  - What % of real Amazon Kindle reviews the model flags as fake
  - Which reviews it is most confident are fake (top suspects)
  - Whether fake-flagged reviews cluster around certain ratings
  - A rating-vs-sentiment mismatch check (e.g. 5-star review that reads negative)

Run after both models are trained:
    !python test_on_kindle.py
"""

import pandas as pd
import numpy as np
import joblib

# ----------------------------
# Load both models
# ----------------------------
DETECTOR_DIR = "./fake_review_detector"
fraud_tfidf   = joblib.load(f"{DETECTOR_DIR}/tfidf.pkl")
fraud_model   = joblib.load(f"{DETECTOR_DIR}/xgb_model.pkl")
sent_tfidf    = joblib.load(f"{DETECTOR_DIR}/tfidf_sentiment.pkl")
sent_model    = joblib.load(f"{DETECTOR_DIR}/sentiment_model.pkl")

SENT_MAP = {0: "negative", 1: "neutral", 2: "positive"}

# ----------------------------
# Load Kindle reviews
# ----------------------------
df = pd.read_csv("preprocessed_kindle_review_.csv").dropna(subset=["reviewText", "rating"])
df["summary"]   = df["summary"].fillna("")
df["full_text"] = df["reviewText"] + " " + df["summary"]
df["text_with_rating"] = df["rating"].astype(str) + " stars. " + df["reviewText"]

print(f"Loaded {len(df)} Kindle reviews\n")

# ----------------------------
# Run fraud model
# ----------------------------
print("Running fraud detection...")
fraud_vecs   = fraud_tfidf.transform(df["text_with_rating"])
fraud_preds  = fraud_model.predict(fraud_vecs)
fraud_probs  = fraud_model.predict_proba(fraud_vecs)[:, 1]  # prob of fake

df["fraud_label"]    = ["fake" if p == 1 else "genuine" for p in fraud_preds]
df["fake_prob"]      = fraud_probs.round(4)

# ----------------------------
# Run sentiment model
# ----------------------------
print("Running sentiment analysis...")
sent_vecs  = sent_tfidf.transform(df["full_text"])
sent_preds = sent_model.predict(sent_vecs)
sent_probs = sent_model.predict_proba(sent_vecs)

df["sentiment"]      = [SENT_MAP[p] for p in sent_preds]
df["sentiment_conf"] = sent_probs.max(axis=1).round(4)

# ----------------------------
# Mismatch detection
# Rating vs sentiment mismatch = extra fraud signal
# e.g. 5-star review predicted as negative sentiment
# ----------------------------
def rating_to_expected_sentiment(r):
    if r <= 2: return "negative"
    if r == 3: return "neutral"
    return "positive"

df["expected_sentiment"] = df["rating"].apply(rating_to_expected_sentiment)
df["sentiment_mismatch"] = df["sentiment"] != df["expected_sentiment"]

# ----------------------------
# Summary Stats
# ----------------------------
print("\n" + "=" * 60)
print("FRAUD DETECTION RESULTS ON KINDLE REVIEWS")
print("=" * 60)
fake_pct = (df["fraud_label"] == "fake").mean() * 100
print(f"  Flagged as FAKE   : {(df['fraud_label']=='fake').sum():,} ({fake_pct:.1f}%)")
print(f"  Flagged as GENUINE: {(df['fraud_label']=='genuine').sum():,} ({100-fake_pct:.1f}%)")

print("\nFake % by star rating:")
print(df.groupby("rating")["fraud_label"].apply(lambda x: f"{(x=='fake').mean()*100:.1f}%"))

print("\n" + "=" * 60)
print("SENTIMENT RESULTS")
print("=" * 60)
print(df["sentiment"].value_counts().to_string())
print(f"\nRating–Sentiment Mismatches: {df['sentiment_mismatch'].sum():,} ({df['sentiment_mismatch'].mean()*100:.1f}%)")

print("\nMismatch % by star rating:")
print(df.groupby("rating")["sentiment_mismatch"].apply(lambda x: f"{x.mean()*100:.1f}%"))

# ----------------------------
# Top 10 most suspicious reviews
# ----------------------------
print("\n" + "=" * 60)
print("TOP 10 MOST SUSPICIOUS REVIEWS (highest fake probability)")
print("=" * 60)
top_suspects = df.nlargest(10, "fake_prob")[
    ["rating", "fake_prob", "sentiment", "sentiment_mismatch", "reviewText"]
]
for i, row in top_suspects.iterrows():
    mismatch_flag = " ⚠ MISMATCH" if row["sentiment_mismatch"] else ""
    print(f"\n[{i}] ★{row['rating']}  fake_prob={row['fake_prob']:.2f}  sentiment={row['sentiment']}{mismatch_flag}")
    print(f"    {str(row['reviewText'])[:120]}...")

# ----------------------------
# Save full results
# ----------------------------
out_path = "./kindle_audit_results.csv"
df[["rating", "reviewText", "fraud_label", "fake_prob",
    "sentiment", "sentiment_conf", "sentiment_mismatch"]].to_csv(out_path, index=False)
print(f"\n\nFull results saved to {out_path}")
