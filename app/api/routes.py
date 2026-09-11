"""API endpoints; business logic remains in services."""
from fastapi import APIRouter, Request
from api.schemas import CustomerInput, HealthResponse, PredictionResponse
from core.exceptions import ModelUnavailableError
from services.risk import classify_risk

router = APIRouter(tags=["churn"])

@router.get("/health", response_model=HealthResponse)
def health(request: Request) -> HealthResponse:
    """Return application readiness and model availability."""
    predictor = request.app.state.predictor
    return HealthResponse(status="ok" if predictor.is_loaded else "degraded", model_loaded=predictor.is_loaded, model_source=predictor.model_source)

@router.post("/predict", response_model=PredictionResponse)
def predict(customer: CustomerInput, request: Request) -> PredictionResponse:
    """Estimate one customer's churn risk."""
    predictor = request.app.state.predictor
    if not predictor.is_loaded:
        raise ModelUnavailableError("Prediction model is not loaded")
    probability = predictor.predict_probability(customer)
    return PredictionResponse(customer_id=customer.customer_id, churn_probability=round(probability, 4), risk_level=classify_risk(probability))

@router.get("/customers", response_model=list[PredictionResponse])
def customers() -> list[PredictionResponse]:
    """Return persisted predictions when a persistence service is added."""
    return []

