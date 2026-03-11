"use client";

import { useEffect, useState } from "react";

interface AccuracyData {
  total_evaluated: number;
  outcome_accuracy?: number;
  exact_score_accuracy?: number;
  over_under_accuracy?: number;
  btts_accuracy?: number;
  message?: string;
}

function StatCard({ label, value, description }: { label: string; value: string; description: string }) {
  return (
    <div className="bg-slate-900 border border-slate-800 rounded-xl p-5">
      <p className="text-sm text-slate-400">{label}</p>
      <p className="text-3xl font-bold text-white mt-1">{value}</p>
      <p className="text-xs text-slate-500 mt-2">{description}</p>
    </div>
  );
}

export default function AccuracyPage() {
  const [data, setData] = useState<AccuracyData | null>(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    fetch("/api/accuracy")
      .then((res) => res.json())
      .then((d) => {
        setData(d);
        setLoading(false);
      })
      .catch(() => setLoading(false));
  }, []);

  if (loading) {
    return <div className="text-center py-12 text-slate-400">Loading accuracy data...</div>;
  }

  if (!data || data.total_evaluated === 0) {
    return (
      <div>
        <h1 className="text-2xl font-bold text-white mb-2">Model Accuracy</h1>
        <div className="bg-slate-900 border border-slate-800 rounded-xl p-8 text-center">
          <p className="text-slate-400">No predictions have been evaluated yet.</p>
          <p className="text-sm text-slate-500 mt-2">
            Accuracy stats will appear after matches with predictions are completed.
          </p>
        </div>
      </div>
    );
  }

  const fmt = (v: number | undefined) => (v !== undefined ? `${(v * 100).toFixed(1)}%` : "N/A");

  return (
    <div>
      <div className="mb-6">
        <h1 className="text-2xl font-bold text-white">Model Accuracy</h1>
        <p className="text-sm text-slate-400 mt-1">
          Tracking our prediction performance across {data.total_evaluated} evaluated matches
        </p>
      </div>

      <div className="grid gap-4 sm:grid-cols-2 lg:grid-cols-4">
        <StatCard
          label="Match Outcome (1X2)"
          value={fmt(data.outcome_accuracy)}
          description="Correctly predicted winner or draw"
        />
        <StatCard
          label="Exact Score"
          value={fmt(data.exact_score_accuracy)}
          description="Predicted the correct scoreline"
        />
        <StatCard
          label="Over/Under 2.5"
          value={fmt(data.over_under_accuracy)}
          description="Correctly predicted total goals"
        />
        <StatCard
          label="BTTS"
          value={fmt(data.btts_accuracy)}
          description="Both teams to score prediction"
        />
      </div>

      <div className="mt-6 bg-slate-900 border border-slate-800 rounded-xl p-5">
        <h2 className="text-lg font-semibold text-white mb-3">How Our Model Works</h2>
        <div className="space-y-2 text-sm text-slate-400">
          <p>
            We use a <strong className="text-slate-300">Dixon-Coles model</strong> — a proven
            statistical approach that estimates attack and defense strengths for each EPL team.
          </p>
          <p>
            The model accounts for home advantage and adjusts for low-scoring game correlations.
            Recent matches are weighted more heavily to capture current form.
          </p>
          <p>
            From these team ratings, we generate a probability matrix for every possible scoreline,
            then derive outcome, exact score, over/under, and BTTS predictions.
          </p>
        </div>
      </div>
    </div>
  );
}
