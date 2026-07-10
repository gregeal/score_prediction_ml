const DEFAULT_LOCAL_API_BASE_URL = "http://localhost:8000";

function trimTrailingSlash(value: string): string {
  return value.replace(/\/+$/, "");
}

export function getApiBaseUrl(): string {
  return trimTrailingSlash(process.env.NEXT_PUBLIC_API_BASE_URL || DEFAULT_LOCAL_API_BASE_URL);
}

export function apiUrl(path: string): string {
  const normalizedPath = path.startsWith("/") ? path : `/${path}`;
  return `${getApiBaseUrl()}${normalizedPath}`;
}

/**
 * Parse an API datetime as UTC even when the backend omits the offset.
 * JavaScript's Date constructor treats offset-less ISO strings as LOCAL
 * time, which would shift every displayed kickoff by the viewer's UTC
 * offset.
 */
export function parseApiDate(value: string): Date {
  const hasOffset = /Z$|[+-]\d{2}:?\d{2}$/.test(value);
  return new Date(hasOffset ? value : `${value}Z`);
}
