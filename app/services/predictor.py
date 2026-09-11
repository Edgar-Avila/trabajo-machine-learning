"""Model loading and inference service."""
import json
import logging
from pathlib import Path
from typing import Any
import joblib
import pandas as pd
from api.schemas import CustomerInput
from core.config import Settings
from core.exceptions import ModelUnavailableError, PredictionError

logger = logging.getLogger(__name__)
DEFAULT_FEATURES = ["Age", "Gender", "Tenure", "Usage Frequency", "Support Calls", "Payment Delay", "Subscription Type", "Contract Length", "Total Spend", "Last Interaction"]

class FallbackChurnModel:
    """Small deterministic demo model; never use this for production decisions."""
    def predict_proba(self, features: pd.DataFrame) -> list[list[float]]:
        """Return a plausible probability while waiting for the trained artifact."""
        row = features.iloc[0]
        score = (0.06 * min(float(row["Support Calls"]), 10)
                 + 0.015 * min(float(row["Payment Delay"]), 30)
                 + 0.02 * min(float(row["Last Interaction"]), 30)
                 - 0.02 * min(float(row["Tenure"]), 24)
                 - 0.005 * min(float(row["Usage Frequency"]), 50))
        probability = max(0.02, min(0.98, 0.25 + score))
        return [[1 - probability, probability]]

class PredictorService:
    """Load a classifier exposing predict_proba and use it for inference."""
    def __init__(self, settings: Settings, model: Any | None = None) -> None:
        self.settings, self.model = settings, model
        self.model_source = "injected" if model is not None else "unavailable"
        self.feature_names = self._load_feature_names(settings.feature_list_path)
        if model is None:
            self._load_model()

    @property
    def is_loaded(self) -> bool:
        """Indicate whether a classifier is ready."""
        return self.model is not None

    def _load_feature_names(self, path: Path | None) -> list[str]:
        if path is None or not path.exists():
            return DEFAULT_FEATURES
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            names = data["features"] if isinstance(data, dict) else data
            if not isinstance(names, list) or not all(isinstance(name, str) for name in names):
                raise ValueError("feature list must contain strings")
            return names
        except (OSError, ValueError, json.JSONDecodeError) as exc:
            raise ModelUnavailableError(f"Could not read feature list at {path}: {exc}") from exc

    def _load_model(self) -> None:
        try:
            self.model = joblib.load(self.settings.model_path)
            self.model_source = "artifact"
            logger.info("Model loaded from %s", self.settings.model_path)
        except (OSError, ValueError, ImportError) as exc:
            if not self.settings.allow_fallback_model:
                raise ModelUnavailableError(f"Model could not be loaded from {self.settings.model_path}") from exc
            self.model, self.model_source = FallbackChurnModel(), "fallback"
            logger.warning("Using fallback model because artifact could not load: %s", exc)

    def predict_probability(self, customer: CustomerInput) -> float:
        """Return the positive-class probability for a validated customer."""
        if self.model is None:
            raise ModelUnavailableError("Prediction model is not loaded")
        payload = customer.model_dump(by_alias=True)
        payload.pop("CustomerID")
        try:
            frame = pd.DataFrame([{name: payload[name] for name in self.feature_names}])
            probability = float(self.model.predict_proba(frame)[0][1])
        except (KeyError, AttributeError, IndexError, TypeError, ValueError) as exc:
            raise PredictionError("The model could not generate a valid churn prediction") from exc
        if not 0 <= probability <= 1:
            raise PredictionError("The model returned a probability outside 0 to 1")
        return probability

