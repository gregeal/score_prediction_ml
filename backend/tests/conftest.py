"""Test configuration — ensures tests use SQLite, not PostgreSQL."""

import os

# Override DATABASE_URL before any app imports touch it
os.environ["DATABASE_URL"] = "sqlite:///./test.db"
