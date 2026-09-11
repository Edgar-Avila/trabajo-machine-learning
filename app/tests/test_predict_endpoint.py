def test_predict_returns_risk_for_valid_customer(client, sample_customer):
    response = client.post("/predict", json=sample_customer)
    assert response.status_code == 200
    assert response.json() == {"customer_id": "C-100", "churn_probability": 0.78, "risk_level": "High"}
def test_predict_rejects_invalid_customer(client, sample_customer):
    sample_customer["Age"] = 12
    response = client.post("/predict", json=sample_customer)
    assert response.status_code == 422
    assert response.json()["detail"] == "Input validation failed"

