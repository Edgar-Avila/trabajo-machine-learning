"""Environment configuration."""
from functools import lru_cache
from pathlib import Path
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

class Settings(BaseSettings):
    """Settings loaded from environment variables and optionally .env."""
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")
    app_name: str = "Customer Churn Risk App"
    app_version: str = "1.0.0"
    model_path: Path = Field(default=Path(__file__).resolve().parents[2] / "model" / "model.pkl")
    model_versions_dir: Path | None = None
    db_path: Path | None = None
    feature_list_path: Path | None = None
    allow_fallback_model: bool = True
    frontend_directory: Path = Path(__file__).resolve().parents[1] / "frontend"
    log_level: str = "INFO"

@lru_cache
def get_settings() -> Settings:
    """Return cached process-level settings."""
    return Settings()

