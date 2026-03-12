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
