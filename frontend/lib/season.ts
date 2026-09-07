export function currentSeasonYear(now: Date = new Date()): number {
  return now.getUTCMonth() >= 6 ? now.getUTCFullYear() : now.getUTCFullYear() - 1;
}

export function seasonLabel(year: number): string {
  return `${year}/${String(year + 1).slice(-2)}`;
}

export function assertCurrentSeason(actual: unknown, expected: number): void {
  if (String(actual) !== String(expected)) {
    throw new Error(
      `The API has not provided the current ${seasonLabel(expected)} season. ` +
      "The backend needs to be deployed and its data pipeline refreshed. Previous-season records are not shown as current.",
    );
  }
}
