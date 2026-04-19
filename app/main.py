from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from app.pipeline import clean_and_engineer
from app.model import predict
from app.llm import get_reasoning

app = FastAPI(
    title="Return & Refund Abuse Detection API",
    description="ML + LLM pipeline to detect abusive return behaviour",
    version="1.0.0"
)


# --- Request schema ---
class OrderRequest(BaseModel):
    Product_Price:    float
    Discount_Applied: float
    Order_Quantity:   int
    Days_to_Return:   float
    Return_Status:    str   # "Returned" or "Not Returned"
    Return_Reason:    str   # e.g. "Changed mind", "Defective", etc.
    User_Age:         int


# --- Response schema ---
class PredictionResponse(BaseModel):
    verdict:     str    # "ABUSE" or "NOT ABUSE"
    probability: str    # e.g. "87.3%"
    abuse_score: int    # 0–5
    signals: dict       # which signals fired
    reasoning: str      # LLM explanation


# --- Health check ---
@app.get("/")
def root():
    return {"status": "ok", "message": "Abuse Detection API is running"}


# --- Main prediction endpoint ---
@app.post("/predict", response_model=PredictionResponse)
def predict_abuse(order: OrderRequest):
    try:
        # Step 1: clean and engineer features
        record = clean_and_engineer(order.model_dump())

        # Step 2: run RF model
        prediction, probability = predict(record)

        # Step 3: get LLM reasoning
        reasoning = get_reasoning(record, prediction, probability)

        # Step 4: build response
        return PredictionResponse(
            verdict     = "ABUSE" if prediction == 1 else "NOT ABUSE",
            probability = f"{probability:.1%}",
            abuse_score = record['abuse_score'],
            signals     = {
                "changed_mind":   bool(record['signal_changed_mind']),
                "high_discount":  bool(record['signal_high_discount']),
                "late_return":    bool(record['signal_late_return']),
                "bulk_return":    bool(record['signal_bulk_return']),
                "expensive_item": bool(record['signal_expensive_item']),
            },
            reasoning = reasoning
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))