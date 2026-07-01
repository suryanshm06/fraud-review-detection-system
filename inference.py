"""
Inference — Combined fraud + sentiment detector
Usage:
    from inference import ReviewAnalyzer
    analyzer = ReviewAnalyzer()
    analyzer.analyze("Absolutely loved this product!", rating=5)
"""

import joblib

SENT_MAP = {0: "negative", 1: "neutral", 2: "positive"}

class ReviewAnalyzer:
    def __init__(self, model_dir="./fake_review_detector"):
        self.fraud_tfidf  = joblib.load(f"{model_dir}/tfidf.pkl")
        self.fraud_model  = joblib.load(f"{model_dir}/xgb_model.pkl")
        self.sent_tfidf   = joblib.load(f"{model_dir}/tfidf_sentiment.pkl")
        self.sent_model   = joblib.load(f"{model_dir}/sentiment_model.pkl")

    def analyze(self, text: str, rating: float = None) -> dict:
        # Fraud prediction
        fraud_text = f"{rating} stars. {text}" if rating else text
        fvec       = self.fraud_tfidf.transform([fraud_text])
        fake_prob  = float(self.fraud_model.predict_proba(fvec)[0][1])
        is_fake    = fake_prob > 0.5

        # Sentiment prediction
        svec       = self.sent_tfidf.transform([text])
        sent_pred  = int(self.sent_model.predict(svec)[0])
        sent_probs = self.sent_model.predict_proba(svec)[0]
        sentiment  = SENT_MAP[sent_pred]

        # Mismatch check
        expected = None
        mismatch = False
        if rating:
            if rating <= 2:   expected = "negative"
            elif rating == 3: expected = "neutral"
            else:             expected = "positive"
            mismatch = (sentiment != expected)

        # Combined suspicion score (fake_prob + 0.2 boost if mismatch)
        suspicion_score = min(1.0, fake_prob + (0.15 if mismatch else 0.0))

        return {
            "fraud": {
                "label":      "fake" if is_fake else "genuine",
                "fake_prob":  round(fake_prob, 4),
            },
            "sentiment": {
                "label":      sentiment,
                "confidence": round(float(max(sent_probs)), 4),
            },
            "rating_sentiment_mismatch": mismatch,
            "suspicion_score": round(suspicion_score, 4),  # 0=clean, 1=very suspicious
        }

    def analyze_batch(self, texts: list, ratings: list = None) -> list:
        ratings = ratings or [None] * len(texts)
        return [self.analyze(t, r) for t, r in zip(texts, ratings)]


if __name__ == "__main__":
    analyzer = ReviewAnalyzer()

    samples = [
        ("This product is okay. Does what it says, nothing special.", 3.0),
        ("Stopped working after 2 days. Very disappointed.", 5.0),   # mismatch!
        ("Absolutely love it, works perfectly, highly recommend!", 5.0),
        ("Worst purchase ever. Complete waste of money.", 1.0),
    ]

    print(f"{'TEXT':<50} {'FRAUD':<10} {'SENTIMENT':<10} {'MISMATCH':<10} {'SUSPICION'}")
    print("-" * 100)
    for text, rating in samples:
        r = analyzer.analyze(text, rating)
        mismatch = "⚠ YES" if r["rating_sentiment_mismatch"] else "no"
        print(
            f"{text[:49]:<50} "
            f"{r['fraud']['label']:<10} "
            f"{r['sentiment']['label']:<10} "
            f"{mismatch:<10} "
            f"{r['suspicion_score']:.2f}"
        )
