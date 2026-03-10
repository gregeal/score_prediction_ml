from sqlalchemy import Column, Integer, Float, String, DateTime, func

from app.models.base import Base


class Prediction(Base):
    __tablename__ = "predictions"

    id = Column(Integer, primary_key=True)
    match_api_id = Column(Integer, nullable=False)
    home_team = Column(String, nullable=False)
    away_team = Column(String, nullable=False)
    predicted_home_goals = Column(Float, nullable=False)
    predicted_away_goals = Column(Float, nullable=False)
    home_win_prob = Column(Float, nullable=False)
    draw_prob = Column(Float, nullable=False)
    away_win_prob = Column(Float, nullable=False)
    over25_prob = Column(Float, nullable=False)
    btts_prob = Column(Float, nullable=False)
    most_likely_score = Column(String)
    confidence = Column(String)
    created_at = Column(DateTime, server_default=func.now())
