from sqlalchemy import Column, DateTime, Float, Integer, String, func

from app.models.base import Base


class MarketOdds(Base):
    __tablename__ = "market_odds"

    id = Column(Integer, primary_key=True)
    match_api_id = Column(Integer, nullable=False, index=True)
    source = Column(String, nullable=False, default="sports-betting")
    captured_at = Column(DateTime, nullable=False, server_default=func.now())
    home_win_odds = Column(Float, nullable=True)
    draw_odds = Column(Float, nullable=True)
    away_win_odds = Column(Float, nullable=True)
    over25_odds = Column(Float, nullable=True)
    under25_odds = Column(Float, nullable=True)
    btts_yes_odds = Column(Float, nullable=True)
    btts_no_odds = Column(Float, nullable=True)
