"""One UTC season policy shared by ingestion and the public API."""

from datetime import datetime, timezone


def utc_now() -> datetime:
    """Naive UTC matches the database's DateTime columns on both supported engines."""
    return datetime.now(timezone.utc).replace(tzinfo=None)


def current_season_year(now: datetime | None = None) -> int:
    now = now or utc_now()
    if now.tzinfo is not None:
        now = now.astimezone(timezone.utc)
    return now.year if now.month >= 7 else now.year - 1


def season_years() -> list[int]:
    year = current_season_year()
    return list(range(year - 4, year + 1))
