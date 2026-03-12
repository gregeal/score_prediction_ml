from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api import fixtures, predictions
from app.config import settings

app = FastAPI(
    title="PredictEPL",
    description="ML-powered EPL score predictions for the Nigerian market",
    version="0.1.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_allowed_origins_list,
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(predictions.router, prefix="/api")
app.include_router(fixtures.router, prefix="/api")


@app.get("/")
def root():
    return {"app": "PredictEPL", "version": "0.1.0", "status": "running"}


@app.get("/health")
def health():
    return {"status": "ok"}
