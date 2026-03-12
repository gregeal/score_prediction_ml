"use client";

import { type ReactNode, useEffect, useState } from "react";

interface SummaryData {
  active_model?: string | null;
  calibrated?: boolean;
  brier_score?: number;
  avg_log_loss?: number;
  model_version?: string | null;
  calibration_version?: string | null;
  benchmark_delta_vs_naive?: number | null;
  evaluation_source?: string | null;
}

interface CalibrationBucket {
  label: string;
  range_start: number;
  range_end: number;
  avg_confidence: number;
  actual_rate: number;
  count: number;
}

interface SegmentMetric {
  name: string;
  count: number;
  outcome_accuracy: number;
  brier_score: number;
  avg_log_loss: number;
}

interface RollingWindowMetric {
  label: string;
  match_count: number;
  outcome_accuracy: number;
  brier_score: number;
  avg_log_loss: number;
}

interface BenchmarkMetric {
  available: boolean;
  total_matches: number;
  outcome_accuracy?: number | null;
  brier_score?: number | null;
  avg_log_loss?: number | null;
}

interface AccuracyData {
  total_evaluated: number;
  outcome_accuracy?: number;
  exact_score_accuracy?: number;
  over_under_accuracy?: number;
  btts_accuracy?: number;
  message?: string;
  summary?: SummaryData;
  calibration?: {
    target?: string;
    buckets?: CalibrationBucket[];
  };
  segments?: SegmentMetric[];
  rolling_backtest?: {
    window_size?: number;
    step_size?: number;
    windows?: RollingWindowMetric[];
  };
  benchmarks?: Record<string, BenchmarkMetric>;
}

function fmtPct(value?: number | null) {
  if (value === undefined || value === null) return "N/A";
  return `${(value * 100).toFixed(1)}%`;
}

function fmtFloat(value?: number | null) {
  if (value === undefined || value === null) return "N/A";
  return value.toFixed(3);
}

function fmtDelta(value?: number | null) {
  if (value === undefined || value === null) return "N/A";
  const sign = value > 0 ? "+" : "";
  return `${sign}${value.toFixed(3)}`;
}

function prettyName(value: string) {
  return value
    .split("_")
    .map((part) => part.charAt(0).toUpperCase() + part.slice(1))
    .join(" ");
}

function StatCard({
  label,
  value,
  description,
  accent = "from-emerald-400/25 to-cyan-400/10",
}: {
  label: string;
  value: string;
  description: string;
  accent?: string;
}) {
  return (
    <div className="relative overflow-hidden rounded-2xl border border-slate-800 bg-slate-950/80 p-5">
      <div className={`absolute inset-x-0 top-0 h-1 bg-gradient-to-r ${accent}`} />
      <p className="text-xs uppercase tracking-[0.2em] text-slate-500">{label}</p>
      <p className="mt-3 text-3xl font-semibold text-white">{value}</p>
      <p className="mt-2 text-sm leading-6 text-slate-400">{description}</p>
    </div>
  );
}

function Section({
  title,
  subtitle,
  children,
}: {
  title: string;
  subtitle: string;
  children: ReactNode;
}) {
  return (
    <section className="rounded-3xl border border-slate-800 bg-slate-950/70 p-6 shadow-[0_30px_80px_-48px_rgba(8,145,178,0.45)]">
      <div className="mb-5">
        <h2 className="text-lg font-semibold text-white">{title}</h2>
        <p className="mt-1 text-sm text-slate-400">{subtitle}</p>
      </div>
      {children}
    </section>
  );
}

export default function AccuracyPage() {
  const [data, setData] = useState<AccuracyData | null>(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    fetch("/api/accuracy")
      .then((res) => res.json())
      .then((payload) => {
        setData(payload);
        setLoading(false);
      })
      .catch(() => setLoading(false));
  }, []);

  if (loading) {
    return <div className="py-12 text-center text-slate-400">Loading accuracy data...</div>;
  }

  if (!data || data.total_evaluated === 0) {
    return (
      <div className="space-y-6">
        <div>
          <h1 className="text-3xl font-semibold text-white">Model Accuracy</h1>
          <p className="mt-2 max-w-2xl text-sm leading-6 text-slate-400">
            This page comes alive once predicted matches have finished. We will show not only whether picks were right,
            but also whether the probabilities were trustworthy.
          </p>
        </div>
        <div className="rounded-3xl border border-slate-800 bg-slate-950/80 p-8 text-center">
          <p className="text-slate-300">No predictions have been evaluated yet.</p>
          <p className="mt-2 text-sm text-slate-500">
            Accuracy stats will appear after matches with stored predictions are completed.
          </p>
        </div>
      </div>
    );
  }

  const summary = data.summary;
  const calibrationBuckets = data.calibration?.buckets ?? [];
  const segments = data.segments ?? [];
  const windows = data.rolling_backtest?.windows ?? [];
  const benchmarks = data.benchmarks ?? {};

  return (
    <div className="space-y-8">
      <div className="rounded-[2rem] border border-slate-800 bg-[radial-gradient(circle_at_top_left,_rgba(16,185,129,0.14),_transparent_34%),radial-gradient(circle_at_top_right,_rgba(34,211,238,0.16),_transparent_30%),rgba(2,6,23,0.85)] p-6">
        <div className="flex flex-col gap-4 lg:flex-row lg:items-end lg:justify-between">
          <div>
            <p className="text-xs uppercase tracking-[0.25em] text-emerald-300/70">Trust Dashboard</p>
            <h1 className="mt-2 text-3xl font-semibold text-white">Model Accuracy</h1>
            <p className="mt-2 max-w-3xl text-sm leading-6 text-slate-300">
              We have evaluated {data.total_evaluated} finished matches. This view tracks not just hit rate, but how
              reliable the probabilities have been over time.
            </p>
          </div>
          <div className="grid gap-2 rounded-2xl border border-white/10 bg-slate-950/40 p-4 text-sm text-slate-300 sm:grid-cols-2">
            <div>
              <p className="text-xs uppercase tracking-[0.2em] text-slate-500">Active Model</p>
              <p className="mt-1 font-medium text-white">{summary?.active_model ?? "Unknown"}</p>
            </div>
            <div>
              <p className="text-xs uppercase tracking-[0.2em] text-slate-500">Calibration</p>
              <p className="mt-1 font-medium text-white">
                {summary?.calibrated ? summary.calibration_version ?? "Enabled" : "Not active"}
              </p>
            </div>
          </div>
        </div>
      </div>

      {data.message && (
        <div className="rounded-2xl border border-cyan-500/20 bg-cyan-500/10 px-5 py-4 text-sm text-cyan-100">
          {data.message}
        </div>
      )}

      <div className="grid gap-4 md:grid-cols-2 xl:grid-cols-4">
        <StatCard
          label="Outcome Accuracy"
          value={fmtPct(data.outcome_accuracy)}
          description="How often the predicted winner or draw matched the final result."
        />
        <StatCard
          label="Brier Score"
          value={fmtFloat(summary?.brier_score)}
          description="Lower is better. This captures the quality of the full 1X2 probability distribution."
          accent="from-cyan-400/25 to-sky-400/10"
        />
        <StatCard
          label="Log Loss"
          value={fmtFloat(summary?.avg_log_loss)}
          description="Lower is better. This punishes confident predictions that miss badly."
          accent="from-amber-400/20 to-orange-400/10"
        />
        <StatCard
          label="Edge vs Naive"
          value={fmtDelta(summary?.benchmark_delta_vs_naive)}
          description="Positive means our Brier score beat the rolling league-prior baseline."
          accent="from-pink-400/25 to-rose-400/10"
        />
      </div>

      <div className="grid gap-4 md:grid-cols-3">
        <StatCard
          label="Exact Score"
          value={fmtPct(data.exact_score_accuracy)}
          description="Correct scoreline hit rate."
          accent="from-slate-500/30 to-slate-300/10"
        />
        <StatCard
          label="Over / Under 2.5"
          value={fmtPct(data.over_under_accuracy)}
          description="Outcome quality for the total-goals market."
          accent="from-teal-400/25 to-emerald-400/10"
        />
        <StatCard
          label="BTTS"
          value={fmtPct(data.btts_accuracy)}
          description="How often both-teams-to-score calls land."
          accent="from-indigo-400/20 to-cyan-400/10"
        />
      </div>

      {calibrationBuckets.length > 0 && (
        <Section
          title="Calibration Check"
          subtitle="When our top pick says a match is 60% likely, the realized hit rate should sit close to that level over time."
        >
          <div className="space-y-3">
            {calibrationBuckets.map((bucket) => {
              const confidenceWidth = Math.max(bucket.avg_confidence * 100, 2);
              const actualWidth = Math.max(bucket.actual_rate * 100, 2);
              return (
                <div key={bucket.label} className="rounded-2xl border border-slate-800 bg-slate-900/70 p-4">
                  <div className="flex flex-col gap-3 sm:flex-row sm:items-center sm:justify-between">
                    <div>
                      <p className="text-sm font-medium text-white">{bucket.label}</p>
                      <p className="text-xs text-slate-500">{bucket.count} evaluated picks</p>
                    </div>
                    <div className="grid gap-2 text-sm text-slate-300 sm:text-right">
                      <span>Predicted: {fmtPct(bucket.avg_confidence)}</span>
                      <span>Actual: {fmtPct(bucket.actual_rate)}</span>
                    </div>
                  </div>
                  <div className="mt-4 space-y-2">
                    <div>
                      <div className="mb-1 flex items-center justify-between text-[11px] uppercase tracking-[0.2em] text-slate-500">
                        <span>Predicted confidence</span>
                        <span>{fmtPct(bucket.avg_confidence)}</span>
                      </div>
                      <div className="h-2 overflow-hidden rounded-full bg-slate-800">
                        <div className="h-full rounded-full bg-cyan-400/80" style={{ width: `${confidenceWidth}%` }} />
                      </div>
                    </div>
                    <div>
                      <div className="mb-1 flex items-center justify-between text-[11px] uppercase tracking-[0.2em] text-slate-500">
                        <span>Observed hit rate</span>
                        <span>{fmtPct(bucket.actual_rate)}</span>
                      </div>
                      <div className="h-2 overflow-hidden rounded-full bg-slate-800">
                        <div className="h-full rounded-full bg-emerald-400/80" style={{ width: `${actualWidth}%` }} />
                      </div>
                    </div>
                  </div>
                </div>
              );
            })}
          </div>
        </Section>
      )}

      <div className="grid gap-6 xl:grid-cols-[1.25fr_0.95fr]">
        {windows.length > 0 && (
          <Section
            title="Rolling Backtests"
            subtitle="A moving window view makes it easier to spot whether the model is stabilizing or drifting."
          >
            <div className="space-y-3">
              {windows.map((window, index) => (
                <div key={`${window.label}-${index}`} className="rounded-2xl border border-slate-800 bg-slate-900/70 p-4">
                  <div className="flex flex-col gap-2 sm:flex-row sm:items-center sm:justify-between">
                    <div>
                      <p className="text-sm font-medium text-white">{window.label}</p>
                      <p className="text-xs text-slate-500">{window.match_count} matches in window</p>
                    </div>
                    <div className="flex gap-4 text-sm text-slate-300">
                      <span>Accuracy {fmtPct(window.outcome_accuracy)}</span>
                      <span>Brier {fmtFloat(window.brier_score)}</span>
                    </div>
                  </div>
                  <div className="mt-4 h-2 overflow-hidden rounded-full bg-slate-800">
                    <div
                      className="h-full rounded-full bg-gradient-to-r from-emerald-400 to-cyan-400"
                      style={{ width: `${Math.max(window.outcome_accuracy * 100, 2)}%` }}
                    />
                  </div>
                </div>
              ))}
            </div>
          </Section>
        )}

        {segments.length > 0 && (
          <Section
            title="Confidence Segments"
            subtitle="These slices help us see where the model is earning trust and where it still needs work."
          >
            <div className="grid gap-3">
              {segments.map((segment) => (
                <div key={segment.name} className="rounded-2xl border border-slate-800 bg-slate-900/70 p-4">
                  <div className="flex items-start justify-between gap-4">
                    <div>
                      <p className="text-sm font-medium text-white">{prettyName(segment.name)}</p>
                      <p className="text-xs text-slate-500">{segment.count} matches</p>
                    </div>
                    <div className="text-right text-sm text-slate-300">
                      <p>{fmtPct(segment.outcome_accuracy)}</p>
                      <p className="text-xs text-slate-500">Brier {fmtFloat(segment.brier_score)}</p>
                    </div>
                  </div>
                </div>
              ))}
            </div>
          </Section>
        )}
      </div>

      {Object.keys(benchmarks).length > 0 && (
        <Section
          title="Benchmark Table"
          subtitle="The key question is not whether the model looks clever, but whether it beats simpler alternatives and the market when odds are available."
        >
          <div className="overflow-x-auto">
            <table className="min-w-full border-separate border-spacing-y-3">
              <thead>
                <tr className="text-left text-[11px] uppercase tracking-[0.2em] text-slate-500">
                  <th className="pb-1 pr-4">Benchmark</th>
                  <th className="pb-1 pr-4">Coverage</th>
                  <th className="pb-1 pr-4">Outcome Accuracy</th>
                  <th className="pb-1 pr-4">Brier</th>
                  <th className="pb-1">Log Loss</th>
                </tr>
              </thead>
              <tbody>
                {Object.entries(benchmarks).map(([name, benchmark]) => (
                  <tr key={name} className="rounded-2xl bg-slate-900/70 text-sm text-slate-300">
                    <td className="rounded-l-2xl border-y border-l border-slate-800 px-4 py-3 font-medium text-white">
                      {prettyName(name)}
                    </td>
                    <td className="border-y border-slate-800 px-4 py-3">{benchmark.total_matches}</td>
                    <td className="border-y border-slate-800 px-4 py-3">
                      {benchmark.available ? fmtPct(benchmark.outcome_accuracy) : "Unavailable"}
                    </td>
                    <td className="border-y border-slate-800 px-4 py-3">
                      {benchmark.available ? fmtFloat(benchmark.brier_score) : "Unavailable"}
                    </td>
                    <td className="rounded-r-2xl border-y border-r border-slate-800 px-4 py-3">
                      {benchmark.available ? fmtFloat(benchmark.avg_log_loss) : "Unavailable"}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </Section>
      )}

      <Section
        title="How To Read This"
        subtitle="These numbers work together. A model can have decent hit rate and still be overconfident."
      >
        <div className="grid gap-4 md:grid-cols-3">
          <div className="rounded-2xl border border-slate-800 bg-slate-900/70 p-4">
            <p className="text-sm font-medium text-white">Calibration</p>
            <p className="mt-2 text-sm leading-6 text-slate-400">
              If the predicted confidence and realized hit rate drift apart, the probabilities need calibration even if
              the picks still look decent on the surface.
            </p>
          </div>
          <div className="rounded-2xl border border-slate-800 bg-slate-900/70 p-4">
            <p className="text-sm font-medium text-white">Rolling windows</p>
            <p className="mt-2 text-sm leading-6 text-slate-400">
              These windows show whether recent form is improving or whether the model is slipping as the season evolves.
            </p>
          </div>
          <div className="rounded-2xl border border-slate-800 bg-slate-900/70 p-4">
            <p className="text-sm font-medium text-white">Benchmarks</p>
            <p className="mt-2 text-sm leading-6 text-slate-400">
              The best signal of product value is whether our probabilities consistently beat the naive baseline and,
              when available, bookmaker-implied prices.
            </p>
          </div>
        </div>
      </Section>
    </div>
  );
}
