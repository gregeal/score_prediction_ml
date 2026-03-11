"use client";

import { useEffect, useState } from "react";

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
    high: "bg-green-500/20 text-green-400 border-green-500/30",
    medium: "bg-yellow-500/20 text-yellow-400 border-yellow-500/30",
    low: "bg-red-500/20 text-red-400 border-red-500/30",
  };
  return (
    <span className={`text-xs px-2 py-0.5 rounded-full border ${colors[level] || colors.medium}`}>
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
    <div className="flex h-2 rounded-full overflow-hidden gap-0.5">
      <div
        className="bg-green-500 rounded-l-full"
        style={{ width: `${homeWin * 100}%` }}
        title={`Home: ${(homeWin * 100).toFixed(0)}%`}
      />
      <div
        className="bg-slate-400"
        style={{ width: `${draw * 100}%` }}
        title={`Draw: ${(draw * 100).toFixed(0)}%`}
      />
      <div
        className="bg-blue-500 rounded-r-full"
        style={{ width: `${awayWin * 100}%` }}
        title={`Away: ${(awayWin * 100).toFixed(0)}%`}
      />
    </div>
  );
}

function FixtureCard({ fixture }: { fixture: Fixture }) {
  const pred = fixture.prediction;
  const matchDate = new Date(fixture.date);
  const dateStr = matchDate.toLocaleDateString("en-NG", {
    weekday: "short",
    day: "numeric",
    month: "short",
  });
  const timeStr = matchDate.toLocaleTimeString("en-NG", {
    hour: "2-digit",
    minute: "2-digit",
  });

  return (
    <div className="bg-slate-900 border border-slate-800 rounded-xl p-4 hover:border-slate-700 transition-colors">
      <div className="flex items-center justify-between mb-3">
        <span className="text-xs text-slate-500">
          Matchday {fixture.matchday} &middot; {dateStr} &middot; {timeStr}
        </span>
        {pred && <ConfidenceBadge level={pred.confidence} />}
      </div>

      <div className="flex items-center justify-between mb-3">
        <div className="flex-1">
          <p className="font-semibold text-white">{fixture.home}</p>
        </div>
        {pred ? (
          <div className="px-4 text-center">
            <p className="text-xl font-bold text-green-400">{pred.outcome_score || pred.most_likely_score}</p>
            <p className="text-[10px] text-slate-500 mt-0.5">PREDICTED</p>
          </div>
        ) : (
          <div className="px-4 text-center">
            <p className="text-xl font-bold text-slate-600">vs</p>
          </div>
        )}
        <div className="flex-1 text-right">
          <p className="font-semibold text-white">{fixture.away}</p>
        </div>
      </div>

      {pred && (
        <>
          <ProbabilityBar homeWin={pred.outcome.home_win} draw={pred.outcome.draw} awayWin={pred.outcome.away_win} />
          <div className="flex justify-between text-xs text-slate-400 mt-1.5">
            <span>H {(pred.outcome.home_win * 100).toFixed(0)}%</span>
            <span>D {(pred.outcome.draw * 100).toFixed(0)}%</span>
            <span>A {(pred.outcome.away_win * 100).toFixed(0)}%</span>
          </div>
          <div className="flex gap-3 mt-3 text-xs">
            <span className="bg-slate-800 px-2 py-1 rounded text-slate-300">
              O/U 2.5: <strong>{(pred.over_under_25 * 100).toFixed(0)}%</strong> Over
            </span>
            <span className="bg-slate-800 px-2 py-1 rounded text-slate-300">
              BTTS: <strong>{(pred.btts * 100).toFixed(0)}%</strong> Yes
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
    fetch("/api/fixtures/upcoming")
      .then((res) => {
        if (!res.ok) throw new Error("Failed to fetch fixtures");
        return res.json();
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
    <div>
      <div className="mb-6">
        <h1 className="text-2xl font-bold text-white">Upcoming Matches</h1>
        <p className="text-sm text-slate-400 mt-1">
          AI-powered predictions for the English Premier League
        </p>
      </div>

      {loading && (
        <div className="text-center py-12 text-slate-400">Loading predictions...</div>
      )}

      {error && (
        <div className="text-center py-12">
          <p className="text-red-400">{error}</p>
          <p className="text-sm text-slate-500 mt-2">
            Make sure the backend API is running on port 8000
          </p>
        </div>
      )}

      {!loading && !error && fixtures.length === 0 && (
        <div className="text-center py-12 text-slate-400">
          <p>No upcoming fixtures found.</p>
          <p className="text-sm mt-2">Run the data pipeline to fetch fixtures and generate predictions.</p>
        </div>
      )}

      <div className="grid gap-4 md:grid-cols-2">
        {fixtures.map((fixture) => (
          <FixtureCard key={fixture.match_id} fixture={fixture} />
        ))}
      </div>
    </div>
  );
}
