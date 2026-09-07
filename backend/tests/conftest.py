"""Test configuration — ensures tests use SQLite, not PostgreSQL."""

import os

# Override DATABASE_URL before any app imports touch it
os.environ["DATABASE_URL"] = "sqlite://"
os.environ["MLFLOW_TRACKING_URI"] = ""
os.environ["FOOTBALL_DATA_API_KEY"] = "test-only"
