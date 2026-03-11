from functools import lru_cache

from sqlalchemy import create_engine
from sqlalchemy.orm import DeclarativeBase, sessionmaker

from app.config import settings


@lru_cache
def _make_engine():
    return create_engine(settings.database_url, echo=False)


@lru_cache
def _make_session_local():
    return sessionmaker(bind=_make_engine())


class Base(DeclarativeBase):
    pass


def get_engine():
    return _make_engine()


def get_session_local():
    return _make_session_local()


def get_db():
    session_factory = get_session_local()
    db = session_factory()
    try:
        yield db
    finally:
        db.close()
