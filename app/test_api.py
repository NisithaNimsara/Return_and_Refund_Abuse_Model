import requests
import json

BASE_URL = "http://127.0.0.1:8000"


def print_result(label: str, response: dict):
    print(f"\n{'='*55}")
    print(f"  TEST: {label}")
    print(f"{'='*55}")
    print(f"  Verdict:     {response.get('verdict')}")
    print(f"  Probability: {response.get('probability')}")
    print(f"  Abuse score: {response.get('abuse_score')}/5")
    print(f"  Signals:     {response.get('signals')}")
    print(f"\n  LLM Reasoning:")
    print(f"  {response.get('reasoning')}")
    print()


# -------------------------------------------------------
# TEST 1 — High risk: 3 signals fired, expensive item
# Expected: ABUSE, high probability
# -------------------------------------------------------
def test_high_risk():
    payload = {
        "Product_Price":    420.0,
        "Discount_Applied": 42.0,
        "Order_Quantity":   2,
        "Days_to_Return":   20.0,
        "Return_Status":    "Returned",
        "Return_Reason":    "Changed mind",
        "User_Age":         29
    }
    r = requests.post(f"{BASE_URL}/predict", json=payload)
    assert r.status_code == 200, f"Expected 200, got {r.status_code}"
    result = r.json()
    assert result['verdict'] == "ABUSE", "Expected ABUSE verdict"
    assert result['abuse_score'] >= 2,   "Expected abuse_score >= 2"
    print_result("High risk order (3 signals)", result)


# -------------------------------------------------------
# TEST 2 — Low risk: 0 signals, cheap item, fast return
# Expected: NOT ABUSE, low probability
# -------------------------------------------------------
def test_low_risk():
    payload = {
        "Product_Price":    49.99,
        "Discount_Applied": 10.0,
        "Order_Quantity":   1,
        "Days_to_Return":   5.0,
        "Return_Status":    "Returned",
        "Return_Reason":    "Defective",
        "User_Age":         45
    }
    r = requests.post(f"{BASE_URL}/predict", json=payload)
    assert r.status_code == 200, f"Expected 200, got {r.status_code}"
    result = r.json()
    assert result['verdict'] == "NOT ABUSE", "Expected NOT ABUSE verdict"
    assert result['abuse_score'] == 0,       "Expected abuse_score of 0"
    print_result("Low risk order (0 signals)", result)


# -------------------------------------------------------
# TEST 3 — Edge case: exactly 2 signals (boundary check)
# Expected: ABUSE (threshold is >= 2)
# -------------------------------------------------------
def test_boundary():
    payload = {
        "Product_Price":    380.0,
        "Discount_Applied": 38.0,
        "Order_Quantity":   1,
        "Days_to_Return":   10.0,
        "Return_Status":    "Returned",
        "Return_Reason":    "Not as described",
        "User_Age":         33
    }
    r = requests.post(f"{BASE_URL}/predict", json=payload)
    assert r.status_code == 200, f"Expected 200, got {r.status_code}"
    result = r.json()
    # signal_high_discount + signal_expensive_item = 2 signals
    assert result['abuse_score'] == 2, "Expected exactly 2 signals"
    print_result("Boundary case (exactly 2 signals)", result)


# -------------------------------------------------------
# TEST 4 — Late return: only 1 signal, but very late
# Expected: NOT ABUSE (only 1 signal, below threshold)
# -------------------------------------------------------
def test_late_only():
    payload = {
        "Product_Price":    89.99,
        "Discount_Applied": 5.0,
        "Order_Quantity":   1,
        "Days_to_Return":   120.0,
        "Return_Status":    "Returned",
        "Return_Reason":    "Wrong item",
        "User_Age":         52
    }
    r = requests.post(f"{BASE_URL}/predict", json=payload)
    assert r.status_code == 200, f"Expected 200, got {r.status_code}"
    result = r.json()
    assert result['signals']['late_return'] == True, "Expected late_return signal"
    assert result['abuse_score'] == 1,               "Expected only 1 signal"
    print_result("Late return only (1 signal, below threshold)", result)


# -------------------------------------------------------
# TEST 5 — Corrupt input: negative Days_to_Return
# Pipeline should sanitise this to 0, not crash
# Expected: no 500 error, Days_to_Return treated as 0
# -------------------------------------------------------
def test_corrupt_input():
    payload = {
        "Product_Price":    200.0,
        "Discount_Applied": 20.0,
        "Order_Quantity":   1,
        "Days_to_Return":   -15.0,   # corrupt value
        "Return_Status":    "Returned",
        "Return_Reason":    "Changed mind",
        "User_Age":         40
    }
    r = requests.post(f"{BASE_URL}/predict", json=payload)
    assert r.status_code == 200, f"Pipeline should sanitise bad input, got {r.status_code}"
    result = r.json()
    # signal_late_return must be 0 because Days_to_Return was sanitised to 0
    assert result['signals']['late_return'] == False, "Corrupt days should not trigger late signal"
    print_result("Corrupt input (negative Days_to_Return sanitised)", result)


# -------------------------------------------------------
# Run all tests
# -------------------------------------------------------
if __name__ == "__main__":
    tests = [
        ("High risk",      test_high_risk),
        ("Low risk",       test_low_risk),
        ("Boundary",       test_boundary),
        ("Late only",      test_late_only),
        ("Corrupt input",  test_corrupt_input),
    ]

    passed = 0
    failed = 0

    for name, test_fn in tests:
        try:
            test_fn()
            print(f"  ✅ PASSED: {name}")
            passed += 1
        except AssertionError as e:
            print(f"  ❌ FAILED: {name} — {e}")
            failed += 1
        except Exception as e:
            print(f"  💥 ERROR:  {name} — {e}")
            failed += 1

    print(f"\n{'='*55}")
    print(f"  Results: {passed} passed, {failed} failed")
    print(f"{'='*55}\n")