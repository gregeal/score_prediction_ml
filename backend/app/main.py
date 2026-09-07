from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from app.api import fixtures, predictions
from app.config import settings
from app.models.base import get_engine
from sqlalchemy import text

app = FastAPI(
    title="PredictEPL",
    description="ML-powered EPL score predictions for the Nigerian market",
    version="0.1.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_allowed_origins_list,
    allow_credentials=False,
    allow_methods=["GET"],
    allow_headers=["Accept", "Content-Type"],
)


@app.middleware("http")
async def security_headers(request, call_next):
    response = await call_next(request)
    response.headers["X-Content-Type-Options"] = "nosniff"
    response.headers["X-Frame-Options"] = "DENY"
    response.headers["Referrer-Policy"] = "no-referrer"
    if request.url.path.startswith("/api/"):
        response.headers["Cache-Control"] = "no-store"
    return response

app.include_router(predictions.router, prefix="/api")
app.include_router(fixtures.router, prefix="/api")


@app.get("/")
def root():
    return {"app": "PredictEPL", "version": "0.1.0", "status": "running"}


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/ready")
def ready():
    """Separate database readiness from lightweight process liveness."""
    try:
        with get_engine().connect() as connection:
            connection.execute(text("SELECT 1 FROM matches LIMIT 1"))
    except Exception:
        raise HTTPException(status_code=503, detail="Database not ready") from None
    return {"status": "ready"}
