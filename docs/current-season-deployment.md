# Current-season deployment checklist

Updating GitHub Pages does not update the Render backend or its database. Local
training also does not populate Render's PostgreSQL database.

## Local Docker Site

Docker images contain a snapshot of the code at build time. Pulling new source or
restarting an existing container does not rebuild that image. From the project root:

```sh
docker compose up -d --build --no-deps backend
docker compose exec -T backend python scripts/run_pipeline.py
```

This preserves PostgreSQL and its existing volume. Check
`http://localhost:8000/api/data-status`, then refresh `http://localhost:3000`.
Do not run `docker compose down -v`; that deletes stored data.

## Hosted Site

1. Confirm the Render web service and daily pipeline both track `master` and use
   the same PostgreSQL `DATABASE_URL`. Keep credentials in Render, not in GitHub.
2. Apply the updated Render blueprint, or set the web service's pre-deploy command
   to `python scripts/run_pipeline.py` (working directory `backend`). The blueprint
   now does this automatically on supported paid services. Do not run training in
   an HTTP request. The existing twice-daily cron handles subsequent refreshes.
3. Deploy the latest backend commit. For an existing manually configured service,
   use Render's Shell or trigger the cron to run the pipeline against its database.
4. Check `/api/data-status`. `current_season` and `latest_available_season` should
   match, `current_season_matches` should be populated, and `latest_prediction_at`
   should be recent. A 404 means an older API release; a 503 requires fixing Render's
   deployment/service/database health first. `/ready` checks database readiness.
5. Rebuild GitHub Pages with `NEXT_PUBLIC_API_BASE_URL` pointing to that API. The
   browser now rejects older or missing season metadata rather than labeling it
   current, refreshes on window focus and every five minutes, and offers a retry.

For September 2026 the expected season is 2026/27 (`2026`). Historical accuracy
backtests may intentionally include earlier seasons; they are not current standings.
Do not relabel 2025/26 records, copy a local SQLite database over production, or
change the browser API URL to localhost to work around a hosted API outage.
