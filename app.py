from pathlib import Path

import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

try:
    from app.pipeline import clean_and_engineer
    from app.model import predict
    from app.llm import get_reasoning
except ImportError:
    from pipeline import clean_and_engineer
    from model import predict
    from llm import get_reasoning


BASE_DIR = Path(__file__).resolve().parent


def find_frontend_dir() -> Path:
    """Find the folder that contains index.html for the frontend."""
    candidates = [
        BASE_DIR / "frontend",
        BASE_DIR / "ui",
        BASE_DIR / "web",
        BASE_DIR,
    ]

    for candidate in candidates:
        if (candidate / "index.html").exists():
            return candidate

    raise RuntimeError(
        "Could not find index.html. Put your frontend files in a 'frontend' folder "
        "or in the same directory as app.py."
    )


FRONTEND_DIR = find_frontend_dir()

app = FastAPI(
    title="RefundGuard",
    description="Single entry point for the backend API and frontend UI",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


class OrderRequest(BaseModel):
    Product_Price: float
    Discount_Applied: float
    Order_Quantity: int
    Days_to_Return: float
    Return_Status: str
    Return_Reason: str
    User_Age: int


class PredictionResponse(BaseModel):
    verdict: str
    probability: str
    abuse_score: int
    signals: dict
    reasoning: str


@app.get("/health")
def health_check():
    return {
        "status": "ok",
        "message": "RefundGuard app is running",
        "frontend_dir": str(FRONTEND_DIR),
    }


@app.post("/predict", response_model=PredictionResponse)
def predict_abuse(order: OrderRequest):
    """
    Initial backend connector:
    - receives the frontend payload
    - calls pipeline cleaning/feature engineering
    - calls the trained model
    - calls the LLM explanation
    - returns a single response to the frontend
    """
    try:
        record = clean_and_engineer(order.model_dump())
        prediction, probability = predict(record)
        reasoning = get_reasoning(record, prediction, probability)

        return PredictionResponse(
            verdict="ABUSE" if prediction == 1 else "NOT ABUSE",
            probability=f"{probability:.1%}",
            abuse_score=record["abuse_score"],
            signals={
                "changed_mind": bool(record["signal_changed_mind"]),
                "high_discount": bool(record["signal_high_discount"]),
                "late_return": bool(record["signal_late_return"]),
                "bulk_return": bool(record["signal_bulk_return"]),
                "expensive_item": bool(record["signal_expensive_item"]),
            },
            reasoning=reasoning,
        )
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc

app.mount("/", StaticFiles(directory=str(FRONTEND_DIR), html=True), name="frontend")

if __name__ == "__main__":
    url = "http://127.0.0.1:8000"
    print(f"\nRefundGuard starting...\nOpen: {url}\n")
    uvicorn.run(app, host="127.0.0.1", port=8000)