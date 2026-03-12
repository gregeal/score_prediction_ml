from pathlib import Path

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
