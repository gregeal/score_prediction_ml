from sqlalchemy import Column, Integer, String, DateTime, Enum
import enum

from app.models.base import Base


class MatchStatus(str, enum.Enum):
    SCHEDULED = "SCHEDULED"
    TIMED = "TIMED"
    IN_PLAY = "IN_PLAY"
    PAUSED = "PAUSED"
    FINISHED = "FINISHED"
    POSTPONED = "POSTPONED"
    CANCELLED = "CANCELLED"


class Match(Base):
    __tablename__ = "matches"

    id = Column(Integer, primary_key=True)
    api_id = Column(Integer, unique=True, nullable=False)
    season = Column(String, nullable=False)
    matchday = Column(Integer)
    utc_date = Column(DateTime, nullable=False)
    status = Column(String, default=MatchStatus.SCHEDULED)
    home_team = Column(String, nullable=False)
    away_team = Column(String, nullable=False)
    home_goals = Column(Integer, nullable=True)
    away_goals = Column(Integer, nullable=True)
