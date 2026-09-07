# Security and Operations

## Deployment Boundary

- The public API is read-only. Run ingestion and training only from the CLI or a
  trusted scheduler, never an unauthenticated HTTP endpoint.
- Expose the API through HTTPS. GitHub Pages exports require an HTTPS backend URL
  and cannot run Python or a database. Keep CORS restricted to your own frontend.
- Compose binds PostgreSQL, MLflow, and the API to localhost. Use a reverse proxy
  for public access, with request/concurrency limits and TLS. Do not publicly expose
  MLflow or PostgreSQL. MLflow's security middleware remains enabled.
- Set a strong PostgreSQL password before using Compose. Existing database volumes
  retain their existing credentials; changing an environment variable does not
  rotate a database password. Never remove volumes to resolve a login failure.
- The application container runs as a non-root user and excludes local secrets,
  databases, logs, and trained artifacts from the build context.

## Dependencies and Models

Use a clean Python 3.12 environment and `uv pip sync backend/requirements.lock`.
Installing into an old environment can leave obsolete MLflow/server dependencies
behind. The original `venv/` is not the verified environment; `.venv/` is.
Use Node 22+ and npm 10+ with `npm ci` in `frontend/`. Older npm releases can ignore
security overrides. Regenerate the backend lock with:

```sh
uv pip compile backend/requirements.txt --universal --generate-hashes --output-file backend/requirements.lock
```

CI runs pytest, a Pages production build, `npm audit`, and `pip-audit`. Audit results
are time-sensitive and do not prove the absence of application vulnerabilities.
Next.js was updated beyond its [August 2026 security release](https://nextjs.org/blog/august-2026-security-release).

Models now use an atomic skops bundle with an explicit type allowlist. Legacy pickle
files are never deserialized. Retrain rather than converting untrusted model files.
Keep the model directory writable only by the training identity. Even skops files
must come from trusted sources; see the [skops guidance](https://skops.readthedocs.io/en/stable/persistence.html).
Optional MLflow tracking requires `requirements-mlflow.txt` and a separately secured
tracking server. Do not use `--disable-security-middleware`.

## Data and Evaluation

The UTC season starts in July. The free data provider may deny older seasons; the
pipeline continues with available history, but treats current-season denial as a
failure. Odds outages are explicitly non-fatal. No fake bookmaker data is substituted.

Only predictions timestamped before kickoff and matching the fixture's teams enter
live evaluation/calibration. Forecast history is retained. Timestamp-aligned folds
avoid training on simultaneous kickoffs. The dashboard fallback is uncalibrated
Dixon-Coles, not a claim about a saved challenger's historical performance. Backtest
results are cached per process and invalidated by result/odds changes.

Bookmaker rows are historical reference prices, fetched after publication, not
time-stamped executable offers. Use equal-coverage model/market rows; do not infer
profitable bets from accuracy alone. The promotion flags remain low-history proxies,
not authoritative promotion records. Actual provider xG/shots data is not included.

`/health` is process liveness. `/ready` checks database/schema availability without
exposing connection details. Schedule `python scripts/run_pipeline.py` from `backend/`
against the same database as the API. On Render the cron and web services have
separate filesystems; the dashboard reads database predictions, not cron artifacts.
Back up PostgreSQL and forecast history. Use controlled schema migrations and
distributed job coordination before scaling to multiple pipeline workers.

## Reporting

Do not post API keys, database credentials, or exploit payloads in public issues.
Use GitHub's private vulnerability reporting if enabled, or contact the maintainer
privately with the affected commit and a minimal reproduction.
