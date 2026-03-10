from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    football_data_api_key: str = ""
    database_url: str = "sqlite:///./predictepl.db"

    model_config = {"env_file": ".env"}


settings = Settings()
