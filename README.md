# 🛡️ Return & Refund Abuse Detection

A hybrid ML + LLM pipeline that detects abusive return behaviour in e-commerce orders. A **Random Forest** model flags suspicious transactions based on engineered abuse signals, and a **local LLM (Ollama / LLaMA 3.2)** generates a human-readable explanation for each decision. A **FastAPI** backend and a vanilla-JS frontend tie everything together into a single deployable application.

---

## 📁 Project Structure

```
Return_and_Refund_Abuse_Model-main/
│
├── app.py                          # Entry point — FastAPI app + static file serving
│
├── app/
│   ├── __init__.py
│   ├── main.py                     # API route definitions (alternate entry)
│   ├── pipeline.py                 # Data cleaning & feature engineering
│   ├── model.py                    # Random Forest inference (loads rf_model.pkl)
│   ├── llm.py                      # Ollama prompt builder & LLM client
│   └── test_api.py                 # API integration tests
│
├── frontend/
│   ├── index.html                  # Single-page UI
│   ├── app.js                      # Fetch calls & result rendering
│   └── styles.css                  # UI styles
│
├── notebook_01_data_pipeline.ipynb # EDA, data cleaning, feature engineering
├── notebook_02_ml_model.ipynb      # Model training, evaluation, export
├── notebook_03_llm_reasoning.ipynb # LLM prompt engineering & integration
│
├── ecommerce_returns_synthetic_data.csv  # Raw dataset (10 000 orders, 17 columns)
├── cleaned_data.csv                      # Processed dataset with engineered features
├── rf_model.pkl                          # Trained Random Forest model
└── requirements.txt                      # Python dependencies
```

---

## 🔍 How It Works

### 1. Feature Engineering (`pipeline.py`)

Five binary **abuse signals** are derived from each order:

| Signal | Condition |
|---|---|
| `signal_changed_mind` | Return reason is `"Changed mind"` |
| `signal_high_discount` | Discount > 35% **and** item was returned |
| `signal_late_return` | Days to return > 90 |
| `signal_bulk_return` | Order quantity ≥ 4 **and** item was returned |
| `signal_expensive_item` | Product price > $350 **and** item was returned |

An `abuse_score` (0–5) is the sum of all triggered signals.

### 2. ML Model (`model.py`)

A **Random Forest Classifier** trained on 11 features (the 5 signals + abuse score + 5 raw order fields). Training used an 80/20 stratified split with `class_weight='balanced'` to handle the ~76/24 class imbalance.

### 3. LLM Reasoning (`llm.py`)

After the model predicts, the app sends a structured prompt to a **local Ollama instance** (LLaMA 3.2) and returns a 3–5 sentence professional explanation of why the return was or wasn't flagged, tailored for a fraud review team.

### 4. API (`app.py`)

A single `POST /predict` endpoint accepts an order payload, runs the full pipeline (clean → predict → explain), and returns a structured JSON response.

---

## 🚀 Getting Started

### Prerequisites

- Python 3.9+
- [Ollama](https://ollama.com) installed and running locally
- LLaMA 3.2 pulled in Ollama

```bash
# Install and start Ollama, then pull the model
ollama pull llama3.2
ollama serve
```

### Installation

```bash
# Clone the repository
git clone https://github.com/your-username/Return_and_Refund_Abuse_Model.git
cd Return_and_Refund_Abuse_Model

# Install dependencies
pip install -r requirements.txt
```

### Run the App

```bash
python app.py
```

Open your browser at **http://127.0.0.1:8000**. The frontend UI and API are served from the same port.

---

## 📡 API Reference

### `POST /predict`

**Request body:**

```json
{
  "Product_Price": 420.00,
  "Discount_Applied": 40.0,
  "Order_Quantity": 5,
  "Days_to_Return": 95.0,
  "Return_Status": "Returned",
  "Return_Reason": "Changed mind",
  "User_Age": 28
}
```

**Response:**

```json
{
  "verdict": "ABUSE",
  "probability": "91.3%",
  "abuse_score": 5,
  "signals": {
    "changed_mind": true,
    "high_discount": true,
    "late_return": true,
    "bulk_return": true,
    "expensive_item": true
  },
  "reasoning": "This return exhibits multiple high-risk patterns simultaneously..."
}
```

### `GET /health`

Returns `200 OK` with a simple status message confirming the app is running.

---

## 📊 Dataset

The raw dataset (`ecommerce_returns_synthetic_data.csv`) contains **10,000 synthetic e-commerce orders** with 17 columns:

| Column | Type | Description |
|---|---|---|
| `Order_ID`, `Product_ID`, `User_ID` | String | Identifiers |
| `Order_Date`, `Return_Date` | String | Dates |
| `Product_Category` | String | Item category |
| `Product_Price` | Float | Item price (USD) |
| `Order_Quantity` | Integer | Units ordered |
| `Return_Reason` | String | Customer-stated reason |
| `Return_Status` | String | `Returned` / `Not Returned` |
| `Days_to_Return` | Float | Days between order and return |
| `User_Age`, `User_Gender`, `User_Location` | Mixed | Customer demographics |
| `Payment_Method`, `Shipping_Method` | String | Transaction metadata |
| `Discount_Applied` | Float | Discount percentage |

---

## 📓 Notebooks

The three notebooks document the full development workflow end-to-end:

| Notebook | Purpose |
|---|---|
| `notebook_01_data_pipeline.ipynb` | EDA, data cleaning, duplicate removal, feature engineering |
| `notebook_02_ml_model.ipynb` | Model training, hyperparameter choices, evaluation metrics, export |
| `notebook_03_llm_reasoning.ipynb` | Prompt engineering, Ollama integration, end-to-end testing |

---

## ⚙️ Configuration

| Variable | Location | Default | Description |
|---|---|---|---|
| `OLLAMA_URL` | `app/llm.py` | `http://localhost:11434/api/generate` | Ollama API endpoint |
| `OLLAMA_MODEL` | `app/llm.py` | `llama3.2` | Model name in Ollama |
| `MODEL_PATH` | `app/model.py` | `rf_model.pkl` (project root) | Path to trained model |

---

## 🧪 Running Tests

```bash
python -m pytest app/test_api.py -v
```

---

## 🛠️ Tech Stack

| Layer | Technology |
|---|---|
| Backend API | FastAPI + Uvicorn |
| ML Model | scikit-learn (Random Forest) |
| LLM | Ollama (LLaMA 3.2, local) |
| Data | pandas, NumPy |
| Frontend | Vanilla HTML / CSS / JavaScript |
| Serialisation | joblib (model), Pydantic (API schemas) |
