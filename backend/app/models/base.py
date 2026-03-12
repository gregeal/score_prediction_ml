import logging
from functools import lru_cache

from sqlalchemy import create_engine, inspect, text
from sqlalchemy.orm import DeclarativeBase, sessionmaker

from app.config import settings

logger = logging.getLogger(__name__)


@lru_cache
def _make_engine():
    return create_engine(settings.database_url, echo=False)


@lru_cache
def _make_session_local():
    return sessionmaker(bind=_make_engine())


class Base(DeclarativeBase):
    pass


def _render_add_column_sql(engine, column) -> str:
    """Render a portable ADD COLUMN fragment for a missing nullable column."""
    preparer = engine.dialect.identifier_preparer
    parts = [preparer.quote(column.name), column.type.compile(dialect=engine.dialect)]

    if column.server_default is not None:
        default_arg = str(column.server_default.arg)
        parts.extend(["DEFAULT", default_arg])

    if not column.nullable:
        if column.server_default is None:
            raise ValueError(
                f"Cannot auto-add required column {column.table.name}.{column.name} without a server default"
            )
        parts.append("NOT NULL")

    return " ".join(parts)


def _sync_missing_columns(engine) -> None:
    """Add new model columns to existing tables when create_all cannot alter them."""
    inspector = inspect(engine)
    preparer = engine.dialect.identifier_preparer

    for table in Base.metadata.sorted_tables:
        if not inspector.has_table(table.name):
            continue

        existing_columns = {column["name"] for column in inspector.get_columns(table.name)}
        missing_columns = [column for column in table.columns if column.name not in existing_columns]
        if not missing_columns:
            continue

        quoted_table = preparer.quote(table.name)
        with engine.begin() as conn:
            for column in missing_columns:
                ddl = _render_add_column_sql(engine, column)
                conn.execute(text(f"ALTER TABLE {quoted_table} ADD COLUMN {ddl}"))
                logger.info("Added missing column %s.%s", table.name, column.name)


@lru_cache
def ensure_database_ready() -> None:
    """Create tables and add any missing nullable columns for existing deployments."""
    import app.models  # noqa: F401

    engine = get_engine()
    Base.metadata.create_all(bind=engine)
    _sync_missing_columns(engine)


def get_engine():
    return _make_engine()


def get_session_local():
    return _make_session_local()


def get_db():
    ensure_database_ready()
    session_factory = get_session_local()
    db = session_factory()
    try:
        yield db
    finally:
        db.close()
