"""API endpoints; business logic remains in services."""
from fastapi import APIRouter, HTTPException, Query, Request
from api.schemas import CustomerHistoryResponse, CustomerInput, CustomerListResponse, DashboardSummary, HealthResponse, PredictionResponse
from core.exceptions import ModelUnavailableError
from services.risk import classify_risk

router = APIRouter(tags=["churn"])

@router.get("/health", response_model=HealthResponse)
def health(request: Request) -> HealthResponse:
    """Return application readiness and model availability."""
    predictor = request.app.state.predictor
    return HealthResponse(status="ok" if predictor.is_loaded else "degraded", model_loaded=predictor.is_loaded, model_source=predictor.model_source, model_version=predictor.model_version)

@router.post("/predict", response_model=PredictionResponse)
def predict(customer: CustomerInput, request: Request) -> PredictionResponse:
    """Estimate one customer's churn risk."""
    predictor = request.app.state.predictor
    if not predictor.is_loaded:
        raise ModelUnavailableError("Prediction model is not loaded")
    probability = predictor.predict_probability(customer)
    return PredictionResponse(customer_id=customer.customer_id, churn_probability=round(probability, 4), risk_level=classify_risk(probability))

@router.get("/customers", response_model=CustomerListResponse)
def customers(page: int = Query(1, ge=1), page_size: int = Query(10, ge=1, le=100), request: Request = None) -> CustomerListResponse:
    """Return persisted predictions for the customer churn table, paginated."""
    repository = request.app.state.customer_repository
    return repository.list_paginated(page, page_size)

@router.get("/customers/summary", response_model=DashboardSummary)
def customers_summary(request: Request) -> DashboardSummary:
    """Return dashboard aggregates over the persisted predictions."""
    return request.app.state.customer_repository.summary()

@router.get("/customers/{customer_id}/history", response_model=CustomerHistoryResponse)
def customer_history(customer_id: str, request: Request) -> CustomerHistoryResponse:
    """Return one customer's churn across model versions."""
    history = request.app.state.customer_repository.history(customer_id)
    if history is None:
        raise HTTPException(status_code=404, detail="No predictions found for this customer")
    return history

