import pytest
from api.schemas import CustomerInput
def test_predictor_uses_mock_model(predictor, sample_customer):
    assert predictor.predict_probability(CustomerInput.model_validate(sample_customer)) == pytest.approx(0.78)

