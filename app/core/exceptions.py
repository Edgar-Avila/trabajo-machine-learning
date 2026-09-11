"""Custom errors and consistent JSON exception handlers."""
import logging
from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse

logger = logging.getLogger(__name__)

class ModelUnavailableError(RuntimeError):
    """Raised when no prediction model can be used."""

class PredictionError(RuntimeError):
    """Raised when a loaded model cannot make a prediction."""

def register_exception_handlers(app: FastAPI) -> None:
    """Attach global JSON error handlers to the app."""
    @app.exception_handler(RequestValidationError)
    async def validation_handler(_: Request, exc: RequestValidationError) -> JSONResponse:
        return JSONResponse(status_code=422, content={"detail": "Input validation failed", "errors": exc.errors()})

    @app.exception_handler(ModelUnavailableError)
    async def model_handler(_: Request, exc: ModelUnavailableError) -> JSONResponse:
        logger.error("Model unavailable: %s", exc)
        return JSONResponse(status_code=500, content={"detail": str(exc)})

    @app.exception_handler(PredictionError)
    async def prediction_handler(_: Request, exc: PredictionError) -> JSONResponse:
        logger.exception("Prediction error")
        return JSONResponse(status_code=500, content={"detail": str(exc)})

    @app.exception_handler(Exception)
    async def unknown_handler(_: Request, exc: Exception) -> JSONResponse:
        logger.exception("Unexpected error")
        return JSONResponse(status_code=500, content={"detail": "An unexpected internal error occurred"})
