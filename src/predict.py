import joblib
import pandas as pd
from pathlib import Path

MODEL_PATH = Path("models/credit_scoring_pipeline.pkl")

def load_model():
    """Load trained pipeline model from disk."""
    return joblib.load(MODEL_PATH)

def predict(new_data: pd.DataFrame):
    """Predict borrower risk given new applicant data."""
    model = load_model()
    preds = model.predict(new_data)
    probs = model.predict_proba(new_data)[:, 1]  # probability of 'yes'
    return preds, probs

if __name__ == "__main__":
    # Example applicant data (must include ALL feature columns)
    new_applicant = pd.DataFrame([{
        "age": 35,
        "job": "admin.",
        "marital": "single",
        "education": "secondary",
        "default": "no",
        "balance": 1200,
        "housing": "yes",
        "loan": "no",
        "contact": "cellular",
        "day": 5,
        "month": "may",
        "duration": 300,
        "campaign": 2,
        "pdays": -1,         # -1 means "never contacted before"
        "previous": 0,
        "poutcome": "unknown"
    }])

    preds, probs = predict(new_applicant)
    print("✅ Prediction:", preds[0])   # 'yes' or 'no'
    print("💡 Default Probability:", round(probs[0], 3))
