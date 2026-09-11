import pytest
from api.schemas import RiskLevel
from services.risk import classify_risk
@pytest.mark.parametrize(("probability","expected"), [(0.0,RiskLevel.LOW),(0.34,RiskLevel.LOW),(0.35,RiskLevel.MEDIUM),(0.64,RiskLevel.MEDIUM),(0.65,RiskLevel.HIGH),(1.0,RiskLevel.HIGH)])
def test_classify_risk_at_all_bands(probability, expected):
    assert classify_risk(probability) == expected
def test_classify_risk_rejects_invalid_probability():
    with pytest.raises(ValueError): classify_risk(1.01)

