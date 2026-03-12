from pathlib import Path

from pydantic_settings import BaseSettings

# Look for .env in backend/ first, then project root
_backend_dir = Path(__file__).resolve().parent.parent
_project_root = _backend_dir.parent
_env_file = _backend_dir / ".env" if (_backend_dir / ".env").exists() else _project_root / ".env"


class Settings(BaseSettings):
    football_data_api_key: str = ""
    database_url: str = "sqlite:///./predictepl.db"
    mlflow_tracking_uri: str = "http://localhost:5000"

    model_config = {"env_file": str(_env_file), "extra": "ignore"}


settings = Settings()
