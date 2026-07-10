from pathlib import Path

from pydantic import field_validator
from pydantic_settings import BaseSettings

# Look for .env in backend/ first, then project root
_backend_dir = Path(__file__).resolve().parent.parent
_project_root = _backend_dir.parent
_env_file = _backend_dir / ".env" if (_backend_dir / ".env").exists() else _project_root / ".env"


class Settings(BaseSettings):
    football_data_api_key: str = ""
    database_url: str = "sqlite:///./predictepl.db"
    mlflow_tracking_uri: str = ""
    cors_allowed_origins: str = "http://localhost:3000,https://gregeal.github.io"

    model_config = {"env_file": str(_env_file), "extra": "ignore"}

    @field_validator("database_url")
    @classmethod
    def _normalize_database_url(cls, value: str) -> str:
        # Render (and Heroku) emit the legacy postgres:// scheme, which
        # SQLAlchemy 2.0 no longer accepts.
        if value.startswith("postgres://"):
            return value.replace("postgres://", "postgresql://", 1)
        return value

    @property
    def cors_allowed_origins_list(self) -> list[str]:
        if self.cors_allowed_origins.strip() == "*":
            return ["*"]
        return [
            origin.strip()
            for origin in self.cors_allowed_origins.split(",")
            if origin.strip()
        ]


settings = Settings()
