import pandas as pd

SIGNAL_COLS = [
    'signal_changed_mind',
    'signal_high_discount',
    'signal_late_return',
    'signal_bulk_return',
    'signal_expensive_item'
]

FEATURE_COLS = [
    'signal_changed_mind', 'signal_high_discount', 'signal_late_return',
    'signal_bulk_return', 'signal_expensive_item', 'abuse_score',
    'Product_Price', 'Discount_Applied', 'Order_Quantity',
    'Days_to_Return', 'User_Age'
]


def clean_and_engineer(raw: dict) -> dict:
    """
    Accepts a raw order dict from the API request.
    Returns a flat dict ready for the RF model.
    """
    # --- Type coercion ---
    record = {
        'Product_Price':    float(raw.get('Product_Price', 0)),
        'Discount_Applied': float(raw.get('Discount_Applied', 0)),
        'Order_Quantity':   int(raw.get('Order_Quantity', 1)),
        'Days_to_Return':   float(raw.get('Days_to_Return', 0)),
        'User_Age':         int(raw.get('User_Age', 0)),
        'Return_Status':    str(raw.get('Return_Status', '')),
        'Return_Reason':    str(raw.get('Return_Reason', 'Not Returned')),
    }

    # --- Corrupt data guard (negative days not allowed) ---
    if record['Days_to_Return'] < 0:
        record['Days_to_Return'] = 0

    # --- Feature engineering (5 abuse signals) ---
    record['signal_changed_mind'] = int(
        record['Return_Reason'] == 'Changed mind'
    )
    record['signal_high_discount'] = int(
        record['Discount_Applied'] > 35 and record['Return_Status'] == 'Returned'
    )
    record['signal_late_return'] = int(
        record['Days_to_Return'] > 90
    )
    record['signal_bulk_return'] = int(
        record['Order_Quantity'] >= 4 and record['Return_Status'] == 'Returned'
    )
    record['signal_expensive_item'] = int(
        record['Product_Price'] > 350 and record['Return_Status'] == 'Returned'
    )

    # --- Abuse score ---
    record['abuse_score'] = sum(record[s] for s in SIGNAL_COLS)

    return record