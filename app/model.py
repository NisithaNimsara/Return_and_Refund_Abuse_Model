import joblib
import pandas as pd
from pathlib import Path

# Load model once when the module is imported
# Path goes up one level from app/ to find rf_model.pkl
MODEL_PATH = Path(__file__).parent.parent / 'rf_model.pkl'

rf_model = joblib.load(MODEL_PATH)

FEATURE_COLS = [
    'signal_changed_mind', 'signal_high_discount', 'signal_late_return',
    'signal_bulk_return', 'signal_expensive_item', 'abuse_score',
    'Product_Price', 'Discount_Applied', 'Order_Quantity',
    'Days_to_Return', 'User_Age'
]


def predict(record: dict) -> tuple[int, float]:
    """
    Takes a cleaned record dict.
    Returns (prediction, probability).
      prediction  : 1 = abuse, 0 = not abuse
      probability : float between 0 and 1
    """
    sample      = pd.DataFrame([{col: record[col] for col in FEATURE_COLS}])
    prediction  = int(rf_model.predict(sample)[0])
    probability = float(rf_model.predict_proba(sample)[0][1])
    return prediction, probability