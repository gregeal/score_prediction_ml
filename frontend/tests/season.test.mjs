import assert from "node:assert/strict";
import test from "node:test";
import { assertCurrentSeason, currentSeasonYear, seasonLabel } from "../lib/season.ts";

test("season changes in July UTC, without a hardcoded end year", () => {
  assert.equal(currentSeasonYear(new Date("2026-06-30T23:59:59Z")), 2025);
  assert.equal(currentSeasonYear(new Date("2026-07-01T00:00:00Z")), 2026);
  assert.equal(currentSeasonYear(new Date("2027-09-01T00:00:00Z")), 2027);
  assert.equal(seasonLabel(2026), "2026/27");
});

test("outdated or unidentified backend seasons cannot masquerade as current", () => {
  for (const value of ["2025", undefined, null, ""]) {
    assert.throws(() => assertCurrentSeason(value, 2026), /current 2026\/27 season/);
  }
  assert.doesNotThrow(() => assertCurrentSeason("2026", 2026));
});
