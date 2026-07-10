"""Tests for application settings."""

from app.config import Settings


class TestDatabaseUrlNormalization:
    def test_legacy_postgres_scheme_is_rewritten(self):
        """Render/Heroku emit postgres://, which SQLAlchemy 2.0 rejects."""
        settings = Settings(database_url="postgres://user:pass@host:5432/db")
        assert settings.database_url == "postgresql://user:pass@host:5432/db"

    def test_modern_scheme_untouched(self):
        settings = Settings(database_url="postgresql://user:pass@host:5432/db")
        assert settings.database_url == "postgresql://user:pass@host:5432/db"

    def test_sqlite_untouched(self):
        settings = Settings(database_url="sqlite:///./predictepl.db")
        assert settings.database_url == "sqlite:///./predictepl.db"
