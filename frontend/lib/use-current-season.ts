"use client";

import { useEffect, useState } from "react";
import { apiFetch } from "./api";
import { assertCurrentSeason, currentSeasonYear, seasonLabel } from "./season";

export function useCurrentSeason<T extends { season: string }>(path: string) {
  const [data, setData] = useState<T | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [attempt, setAttempt] = useState(0);

  useEffect(() => {
    let active = true;
    let fetching = false;
    async function refresh() {
      if (fetching) return;
      fetching = true;
      const year = currentSeasonYear();
      try {
        const response = await apiFetch(`${path}?season=${year}`);
        if (!response.ok) {
          throw new Error(`The prediction API is unavailable (HTTP ${response.status}). Current ${seasonLabel(year)} data cannot be loaded. Please retry shortly.`);
        }
        const payload: T = await response.json();
        assertCurrentSeason(payload.season, year);
        if (active) {
          setData(payload);
          setError(null);
        }
      } catch (reason) {
        if (active) {
          setData(null);
          setError(reason instanceof Error ? reason.message : "Unable to load current-season data.");
        }
      } finally {
        fetching = false;
        if (active) setLoading(false);
      }
    }
    void refresh();
    const timer = window.setInterval(refresh, 5 * 60 * 1000);
    window.addEventListener("focus", refresh);
    return () => {
      active = false;
      window.clearInterval(timer);
      window.removeEventListener("focus", refresh);
    };
  }, [path, attempt]);

  return { data, loading, error, retry: () => { setLoading(true); setAttempt(value => value + 1); } };
}
