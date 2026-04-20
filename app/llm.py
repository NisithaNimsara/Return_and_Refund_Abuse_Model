import requests


OLLAMA_URL = "http://localhost:11434/api/generate"
OLLAMA_MODEL = "llama3.2"


def build_prompt(record: dict, prediction: int, probability: float) -> str:
    fired = []
    if record['signal_changed_mind']   == 1: fired.append("returned with reason 'Changed mind'")
    if record['signal_high_discount']  == 1: fired.append(f"high discount ({record['Discount_Applied']}%) applied then returned")
    if record['signal_late_return']    == 1: fired.append(f"returned very late ({record['Days_to_Return']} days after order)")
    if record['signal_bulk_return']    == 1: fired.append(f"bulk quantity ({record['Order_Quantity']} units) returned")
    if record['signal_expensive_item'] == 1: fired.append(f"high-value item (${record['Product_Price']}) returned")

    fired_text = "\n".join(f"  - {s}" for s in fired) if fired else "  - None"
    verdict    = "ABUSE DETECTED" if prediction == 1 else "LEGITIMATE RETURN"

    return f"""You are a fraud analyst at an e-commerce company.
A machine learning model has reviewed a customer return and made a decision.
Explain this decision in 3-5 sentences for the fraud review team.

--- ORDER DETAILS ---
Product price:    ${record['Product_Price']}
Discount applied: {record['Discount_Applied']}%
Order quantity:   {record['Order_Quantity']}
Days to return:   {record['Days_to_Return']}
Return reason:    {record.get('Return_Reason', 'N/A')}
Customer age:     {record['User_Age']}

--- ABUSE SIGNALS TRIGGERED ---
{fired_text}

--- MODEL DECISION ---
Verdict:     {verdict}
Confidence:  {probability:.1%}
Abuse score: {record['abuse_score']} out of 5

--- YOUR TASK ---
Write a short professional explanation of why this return was flagged or not flagged.
Explain the pattern and business risk. Do not just repeat the numbers.
"""


def get_reasoning(record: dict, prediction: int, probability: float) -> str:
    """
    Sends the prompt to Ollama and returns the LLM explanation in string.
    """
    try:
        response = requests.post(
            OLLAMA_URL,
            json={
                "model":  OLLAMA_MODEL,
                "prompt": build_prompt(record, prediction, probability),
                "stream": False
            },
            timeout=60
        )
        response.raise_for_status()
        return response.json()["response"].strip()

    except requests.exceptions.ConnectionError:
        return "LLM reasoning unavailable — make sure Ollama is running (ollama serve)."
    except Exception as e:
        return f"LLM reasoning error: {str(e)}"