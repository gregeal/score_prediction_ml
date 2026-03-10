from sqlalchemy import Column, Integer, String

from app.models.base import Base


class Team(Base):
    __tablename__ = "teams"

    id = Column(Integer, primary_key=True)
    api_id = Column(Integer, unique=True, nullable=False)
    name = Column(String, nullable=False)
    short_name = Column(String)
    tla = Column(String(3))
    crest_url = Column(String)
