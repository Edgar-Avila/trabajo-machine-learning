"""Risk classification rules."""
from api.schemas import RiskLevel

def classify_risk(probability: float, medium_threshold: float = 0.35, high_threshold: float = 0.65) -> RiskLevel:
    """Map a valid churn probability to Low, Medium, or High risk."""
    if not 0 <= probability <= 1:
        raise ValueError("Probability must be between 0 and 1")
    if not 0 < medium_threshold < high_threshold < 1:
        raise ValueError("Risk thresholds must satisfy 0 < medium < high < 1")
    if probability >= high_threshold:
        return RiskLevel.HIGH
    if probability >= medium_threshold:
        return RiskLevel.MEDIUM
    return RiskLevel.LOW

