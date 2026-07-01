"""
Inference — TF-IDF + XGBoost fake review detector

Usage:
    from inference import ReviewFraudDetector
    detector = ReviewFraudDetector()
    detector.predict("This product is absolutely amazing, best ever!!!")
"""

import joblib

class ReviewFraudDetector:
    def __init__(self, model_dir="./fake_review_detector"):
        self.tfidf = joblib.load(f"{model_dir}/tfidf.pkl")
        self.model = joblib.load(f"{model_dir}/xgb_model.pkl")

    def predict(self, text: str, rating: float = None) -> dict:
        # Prepend rating if provided (matches training format)
        if rating is not None:
            text = f"{rating} stars. {text}"

        vec = self.tfidf.transform([text])
        pred = int(self.model.predict(vec)[0])
        probs = self.model.predict_proba(vec)[0]

        return {
            "label":        "fake" if pred == 1 else "genuine",
            "confidence":   round(float(max(probs)), 4),
            "fake_prob":    round(float(probs[1]), 4),
            "genuine_prob": round(float(probs[0]), 4),
        }

    def predict_batch(self, texts: list, ratings: list = None) -> list:
        if ratings:
            texts = [f"{r} stars. {t}" for r, t in zip(ratings, texts)]
        vecs  = self.tfidf.transform(texts)
        preds = self.model.predict(vecs)
        probs = self.model.predict_proba(vecs)
        return [
            {
                "label":        "fake" if p == 1 else "genuine",
                "confidence":   round(float(max(pr)), 4),
                "fake_prob":    round(float(pr[1]), 4),
                "genuine_prob": round(float(pr[0]), 4),
            }
            for p, pr in zip(preds, probs)
        ]


if __name__ == "__main__":
    detector = ReviewFraudDetector()
    samples = [
        ("Amazing product! Best purchase of my life!!! 5 stars!!! Wow!!!",    5.0),
        ("Works fine. A bit noisy but does the job. Would buy again.",         4.0),
        ("Great great great great amazing amazing must buy now incredible!!!", 5.0),
        ("Stopped working after 2 weeks. Disappointed.",                       1.0),
    ]
    for text, rating in samples:
        result = detector.predict(text, rating)
        print(f"[{result['label'].upper():7s}] conf={result['confidence']:.2f} | {text[:55]}")
