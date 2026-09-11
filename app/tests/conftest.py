import pytest
from fastapi.testclient import TestClient
from core.config import Settings
from main import create_app
from services.predictor import PredictorService
class MockChurnModel:
    def predict_proba(self, _features):
        return [[0.22, 0.78]]
@pytest.fixture
def sample_customer():
    return {"CustomerID":"C-100","Age":35,"Gender":"Female","Tenure":12,"Usage Frequency":15,"Support Calls":2,"Payment Delay":3,"Subscription Type":"Premium","Contract Length":"Annual","Total Spend":650.50,"Last Interaction":4}
@pytest.fixture
def mock_model(): return MockChurnModel()
@pytest.fixture
def predictor(mock_model): return PredictorService(Settings(allow_fallback_model=False), model=mock_model)
@pytest.fixture
def client(predictor): return TestClient(create_app(predictor=predictor))

