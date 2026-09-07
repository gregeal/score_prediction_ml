"""Best-effort telemetry must never interrupt computation or persistence."""

import logging

logger = logging.getLogger(__name__)


class SafeRun:
    def __init__(self, context):
        self.context = context

    def __enter__(self):
        try:
            if self.context:
                self.context.__enter__()
        except Exception:
            logger.warning("MLflow run unavailable; continuing without tracking")
            self.context = None
        return self

    def __exit__(self, exc_type, exc, tb):
        try:
            if self.context:
                self.context.__exit__(exc_type, exc, tb)
        except Exception:
            logger.warning("MLflow run finalization failed")
        return False


class SafeTracking:
    def __init__(self, client):
        self.client = client

    def __getattr__(self, name):
        def call(*args, **kwargs):
            result = None
            try:
                result = getattr(self.client, name)(*args, **kwargs)
            except Exception:
                logger.warning("MLflow %s unavailable; continuing without tracking", name)
            return SafeRun(result) if name == "start_run" else result
        return call
