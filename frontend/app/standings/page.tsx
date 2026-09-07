"use client";

import { useEffect, useState } from "react";

import { apiFetch } from "@/lib/api";

interface TeamStanding {
  position: number;
  team: string;
  played: number;
  won: number;
  drawn: number;
  lost: number;
  goals_for: number;
  goals_against: number;
  goal_difference: number;
  points: number;
}

export default function StandingsPage() {
  const [standings, setStandings] = useState<TeamStanding[]>([]);
  const [season, setSeason] = useState<string | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    apiFetch("/api/standings")
      .then((res) => {
        if (!res.ok) {
          throw new Error(`API responded with status ${res.status}`);
        }
        return res.json();
      })
      .then((data) => {
        setStandings(data.standings ?? []);
        setSeason(data.season ?? null);
        setLoading(false);
      })
      .catch(() => {
        setError("Could not reach the prediction API. Please try again shortly.");
        setLoading(false);
      });
  }, []);

  if (loading) {
    return <div className="text-center py-12 text-slate-400">Loading standings...</div>;
  }

  if (error) {
    return (
      <div className="bg-slate-900 border border-rose-500/30 rounded-xl p-8 text-center text-rose-300">
        {error}
      </div>
    );
  }

  return (
    <div>
      <div className="mb-6">
        <h1 className="text-2xl font-bold text-white">Premier League Table</h1>
        {season && (
          <p className="text-sm text-slate-400 mt-1">
            {season}/{parseInt(season) + 1} Season
          </p>
        )}
      </div>

      {standings.length === 0 ? (
        <div className="bg-slate-900 border border-slate-800 rounded-xl p-8 text-center text-slate-400">
          No data has been synced for this season yet. Refresh the data pipeline to load the current fixtures and results.
        </div>
      ) : (
        <div className="bg-slate-900 border border-slate-800 rounded-xl overflow-hidden">
          <div className="overflow-x-auto">
            <table className="w-full text-sm">
              <thead>
                <tr className="border-b border-slate-800 text-slate-400 text-xs">
                  <th className="text-left py-3 px-3">#</th>
                  <th className="text-left py-3 px-3">Team</th>
                  <th className="text-center py-3 px-2">P</th>
                  <th className="text-center py-3 px-2">W</th>
                  <th className="text-center py-3 px-2">D</th>
                  <th className="text-center py-3 px-2">L</th>
                  <th className="text-center py-3 px-2">GF</th>
                  <th className="text-center py-3 px-2">GA</th>
                  <th className="text-center py-3 px-2">GD</th>
                  <th className="text-center py-3 px-3 font-bold text-slate-300">Pts</th>
                </tr>
              </thead>
              <tbody>
                {standings.map((team) => (
                  <tr
                    key={team.team}
                    className="border-b border-slate-800/50 hover:bg-slate-800/30 transition-colors"
                  >
                    <td className="py-2.5 px-3 text-slate-500">{team.position}</td>
                    <td className="py-2.5 px-3 font-medium text-white">{team.team}</td>
                    <td className="py-2.5 px-2 text-center text-slate-400">{team.played}</td>
                    <td className="py-2.5 px-2 text-center text-slate-400">{team.won}</td>
                    <td className="py-2.5 px-2 text-center text-slate-400">{team.drawn}</td>
                    <td className="py-2.5 px-2 text-center text-slate-400">{team.lost}</td>
                    <td className="py-2.5 px-2 text-center text-slate-400">{team.goals_for}</td>
                    <td className="py-2.5 px-2 text-center text-slate-400">{team.goals_against}</td>
                    <td className="py-2.5 px-2 text-center text-slate-300">
                      {team.goal_difference > 0 ? `+${team.goal_difference}` : team.goal_difference}
                    </td>
                    <td className="py-2.5 px-3 text-center font-bold text-white">{team.points}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>
      )}
    </div>
  );
}
