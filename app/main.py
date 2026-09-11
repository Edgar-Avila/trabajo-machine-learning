"""ASGI entry point."""
import logging
from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from api.routes import router
from core.config import get_settings
from core.exceptions import register_exception_handlers
from services.predictor import PredictorService

def create_app(predictor: PredictorService | None = None) -> FastAPI:
    """Build the application, allowing predictor injection for tests."""
    settings = get_settings()
    logging.basicConfig(level=settings.log_level)
    application = FastAPI(title=settings.app_name, version=settings.app_version)
    application.state.predictor = predictor or PredictorService(settings)
    application.include_router(router)
    register_exception_handlers(application)
    application.mount("/", StaticFiles(directory=settings.frontend_directory, html=True), name="frontend")
    return application

app = create_app()

