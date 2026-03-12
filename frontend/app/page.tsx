"use client";

import Link from "next/link";
import { useEffect, useState } from "react";

import { apiUrl } from "@/lib/api";

interface Prediction {
  outcome: { home_win: number; draw: number; away_win: number };
  most_likely_score: string;
  outcome_score: string;
  over_under_25: number;
  btts: number;
  confidence: string;
}

interface Fixture {
  match_id: number;
  home: string;
  away: string;
  date: string;
  matchday: number;
  prediction: Prediction | null;
}

function ConfidenceBadge({ level }: { level: string }) {
  const colors: Record<string, string> = {
    high: "border-emerald-400/30 bg-emerald-400/15 text-emerald-300",
    medium: "border-amber-400/30 bg-amber-400/15 text-amber-300",
    low: "border-rose-400/30 bg-rose-400/15 text-rose-300",
  };

  return (
    <span className={`rounded-full border px-2 py-0.5 text-[11px] uppercase tracking-[0.15em] ${colors[level] || colors.medium}`}>
      {level}
    </span>
  );
}

function ProbabilityBar({
  homeWin,
  draw,
  awayWin,
}: {
  homeWin: number;
  draw: number;
  awayWin: number;
}) {
  return (
    <div className="flex h-2 overflow-hidden rounded-full bg-slate-800">
      <div
        className="bg-emerald-400"
        style={{ width: `${homeWin * 100}%` }}
        title={`Home: ${(homeWin * 100).toFixed(0)}%`}
      />
      <div
        className="bg-slate-300"
        style={{ width: `${draw * 100}%` }}
        title={`Draw: ${(draw * 100).toFixed(0)}%`}
      />
      <div
        className="bg-cyan-400"
        style={{ width: `${awayWin * 100}%` }}
        title={`Away: ${(awayWin * 100).toFixed(0)}%`}
      />
    </div>
  );
}

function QuickLinkCard({
  href,
  eyebrow,
  title,
  description,
  accent,
}: {
  href: string;
  eyebrow: string;
  title: string;
  description: string;
  accent: string;
}) {
  return (
    <Link
      href={href}
      className="group relative overflow-hidden rounded-[1.75rem] border border-slate-800 bg-slate-950/80 p-5 transition-transform duration-200 hover:-translate-y-1 hover:border-slate-700"
    >
      <div className={`absolute inset-x-0 top-0 h-1 bg-gradient-to-r ${accent}`} />
      <p className="text-[11px] uppercase tracking-[0.22em] text-slate-500">{eyebrow}</p>
      <h2 className="mt-3 text-xl font-semibold text-white">{title}</h2>
      <p className="mt-2 text-sm leading-6 text-slate-400">{description}</p>
      <p className="mt-5 text-sm font-medium text-emerald-300 transition-colors group-hover:text-cyan-300">
        Open view
      </p>
    </Link>
  );
}

function FixtureCard({ fixture }: { fixture: Fixture }) {
  const prediction = fixture.prediction;
  const matchDate = new Date(fixture.date);
  const dateLabel = matchDate.toLocaleDateString("en-NG", {
    weekday: "short",
    day: "numeric",
    month: "short",
  });
  const timeLabel = matchDate.toLocaleTimeString("en-NG", {
    hour: "2-digit",
    minute: "2-digit",
  });

  return (
    <div className="rounded-[1.5rem] border border-slate-800 bg-slate-950/80 p-5 shadow-[0_26px_60px_-40px_rgba(16,185,129,0.35)] transition-colors hover:border-slate-700">
      <div className="mb-4 flex items-center justify-between gap-3">
        <span className="text-xs text-slate-500">
          Matchday {fixture.matchday} | {dateLabel} | {timeLabel}
        </span>
        {prediction && <ConfidenceBadge level={prediction.confidence} />}
      </div>

      <div className="mb-4 flex items-center justify-between gap-4">
        <div className="flex-1">
          <p className="font-semibold text-white">{fixture.home}</p>
        </div>
        {prediction ? (
          <div className="rounded-2xl border border-emerald-400/15 bg-emerald-400/10 px-4 py-2 text-center">
            <p className="text-xl font-bold text-emerald-300">{prediction.outcome_score || prediction.most_likely_score}</p>
            <p className="mt-0.5 text-[10px] uppercase tracking-[0.22em] text-slate-500">Predicted</p>
          </div>
        ) : (
          <div className="px-3 text-center">
            <p className="text-xl font-bold text-slate-600">vs</p>
          </div>
        )}
        <div className="flex-1 text-right">
          <p className="font-semibold text-white">{fixture.away}</p>
        </div>
      </div>

      {prediction && (
        <>
          <ProbabilityBar
            homeWin={prediction.outcome.home_win}
            draw={prediction.outcome.draw}
            awayWin={prediction.outcome.away_win}
          />
          <div className="mt-2 flex justify-between text-xs text-slate-400">
            <span>H {(prediction.outcome.home_win * 100).toFixed(0)}%</span>
            <span>D {(prediction.outcome.draw * 100).toFixed(0)}%</span>
            <span>A {(prediction.outcome.away_win * 100).toFixed(0)}%</span>
          </div>
          <div className="mt-4 flex flex-wrap gap-2 text-xs">
            <span className="rounded-full border border-slate-800 bg-slate-900 px-3 py-1 text-slate-300">
              O/U 2.5: <strong>{(prediction.over_under_25 * 100).toFixed(0)}%</strong> Over
            </span>
            <span className="rounded-full border border-slate-800 bg-slate-900 px-3 py-1 text-slate-300">
              BTTS: <strong>{(prediction.btts * 100).toFixed(0)}%</strong> Yes
            </span>
          </div>
        </>
      )}
    </div>
  );
}

export default function HomePage() {
  const [fixtures, setFixtures] = useState<Fixture[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    fetch(apiUrl("/api/fixtures/upcoming"))
      .then((response) => {
        if (!response.ok) throw new Error("Failed to fetch fixtures");
        return response.json();
      })
      .then((data) => {
        setFixtures(data.fixtures);
        setLoading(false);
      })
      .catch((err) => {
        setError(err.message);
        setLoading(false);
      });
  }, []);

  return (
    <div className="space-y-8">
      <section className="overflow-hidden rounded-[2rem] border border-slate-800 bg-[radial-gradient(circle_at_top_left,_rgba(16,185,129,0.18),_transparent_32%),radial-gradient(circle_at_top_right,_rgba(34,211,238,0.16),_transparent_32%),rgba(2,6,23,0.88)] p-6 md:p-8">
        <div className="grid gap-8 lg:grid-cols-[1.35fr_0.95fr] lg:items-end">
          <div>
            <p className="text-[11px] uppercase tracking-[0.26em] text-emerald-300/80">Prediction Hub</p>
            <h1 className="mt-3 max-w-3xl text-4xl font-semibold leading-tight text-white md:text-5xl">
              Match picks on the left. Trust signals on the right.
            </h1>
            <p className="mt-4 max-w-2xl text-sm leading-7 text-slate-300">
              PredictEPL now separates the headline prediction from the evidence behind it. Use the fixtures list for upcoming
              matches, then jump into the accuracy dashboard to check calibration, rolling backtests, and bookmaker benchmarks.
            </p>
            <div className="mt-6 flex flex-wrap gap-3">
              <Link
                href="/accuracy"
                className="rounded-full bg-emerald-300 px-5 py-2.5 text-sm font-semibold text-slate-950 transition-colors hover:bg-cyan-300"
              >
                Open Accuracy Dashboard
              </Link>
              <Link
                href="/standings"
                className="rounded-full border border-slate-700 bg-slate-950/50 px-5 py-2.5 text-sm font-semibold text-slate-200 transition-colors hover:border-slate-500 hover:bg-slate-900"
              >
                See League Table
              </Link>
            </div>
          </div>

          <div className="grid gap-3 rounded-[1.75rem] border border-white/10 bg-slate-950/45 p-5">
            <div className="rounded-2xl border border-slate-800 bg-slate-950/80 p-4">
              <p className="text-[11px] uppercase tracking-[0.2em] text-slate-500">Calibration</p>
              <p className="mt-2 text-lg font-semibold text-white">Are our 60% calls really landing near 60%?</p>
            </div>
            <div className="rounded-2xl border border-slate-800 bg-slate-950/80 p-4">
              <p className="text-[11px] uppercase tracking-[0.2em] text-slate-500">Rolling Form</p>
              <p className="mt-2 text-lg font-semibold text-white">See whether the model is getting sharper or drifting.</p>
            </div>
            <div className="rounded-2xl border border-slate-800 bg-slate-950/80 p-4">
              <p className="text-[11px] uppercase tracking-[0.2em] text-slate-500">Benchmarks</p>
              <p className="mt-2 text-lg font-semibold text-white">Compare our probabilities against naive priors and bookmaker odds.</p>
            </div>
          </div>
        </div>
      </section>

      <section className="grid gap-4 md:grid-cols-3">
        <QuickLinkCard
          href="/accuracy"
          eyebrow="Trust"
          title="Accuracy Dashboard"
          description="Brier score, log loss, confidence buckets, rolling windows, and segment breakdowns in one place."
          accent="from-emerald-400/40 to-cyan-400/25"
        />
        <QuickLinkCard
          href="/accuracy"
          eyebrow="Benchmarks"
          title="Model vs Market"
          description="See whether the model is beating the rolling league prior and bookmaker implied probabilities."
          accent="from-cyan-400/40 to-sky-400/25"
        />
        <QuickLinkCard
          href="/standings"
          eyebrow="Context"
          title="League Table"
          description="Check the current table alongside predictions to understand the season story behind the probabilities."
          accent="from-amber-400/30 to-emerald-400/20"
        />
      </section>

      <section>
        <div className="mb-5 flex flex-col gap-2 sm:flex-row sm:items-end sm:justify-between">
          <div>
            <p className="text-[11px] uppercase tracking-[0.22em] text-slate-500">Upcoming</p>
            <h2 className="mt-2 text-2xl font-semibold text-white">Upcoming Matches</h2>
            <p className="mt-1 text-sm text-slate-400">
              AI-powered EPL scorelines, probability splits, and confidence flags for the next fixtures.
            </p>
          </div>
          <Link href="/accuracy" className="text-sm font-medium text-emerald-300 transition-colors hover:text-cyan-300">
            View trust dashboard
          </Link>
        </div>

        {loading && <div className="py-12 text-center text-slate-400">Loading predictions...</div>}

        {error && (
          <div className="rounded-[1.5rem] border border-rose-500/20 bg-rose-500/10 px-6 py-10 text-center">
            <p className="text-rose-300">{error}</p>
            <p className="mt-2 text-sm text-slate-400">Make sure the backend API is running on port 8000.</p>
          </div>
        )}

        {!loading && !error && fixtures.length === 0 && (
          <div className="rounded-[1.5rem] border border-slate-800 bg-slate-950/80 px-6 py-12 text-center text-slate-400">
            <p>No upcoming fixtures found.</p>
            <p className="mt-2 text-sm">Run the data pipeline to fetch fixtures, odds, and generate predictions.</p>
          </div>
        )}

        <div className="grid gap-4 md:grid-cols-2">
          {fixtures.map((fixture) => (
            <FixtureCard key={fixture.match_id} fixture={fixture} />
          ))}
        </div>
      </section>
    </div>
  );
}
