import type { NextConfig } from "next";
import { PHASE_DEVELOPMENT_SERVER } from "next/constants";

const repoName = process.env.GITHUB_REPOSITORY?.split("/")[1] || "score_prediction_ml";
const isGithubPagesBuild = process.env.GITHUB_PAGES === "true";
const basePath = isGithubPagesBuild ? `/${repoName}` : "";

// A Pages build without an API base URL would silently bake localhost:8000
// into the static site and ship a fully broken deployment with green CI.
if (isGithubPagesBuild && !process.env.NEXT_PUBLIC_API_BASE_URL) {
  throw new Error(
    "NEXT_PUBLIC_API_BASE_URL must be set for GitHub Pages builds " +
      "(set the NEXT_PUBLIC_API_BASE_URL repository variable to the deployed backend URL).",
  );
}

const nextConfig: NextConfig = {
  output: isGithubPagesBuild ? "export" : undefined,
  outputFileTracingRoot: process.cwd(),
  trailingSlash: isGithubPagesBuild,
  images: {
    unoptimized: true,
  },
  basePath,
  assetPrefix: basePath || undefined,
};

if (isGithubPagesBuild) {
  const api = new URL(process.env.NEXT_PUBLIC_API_BASE_URL!);
  if (api.protocol !== "https:" || ["localhost", "127.0.0.1", "[::1]"].includes(api.hostname) || api.username || api.password) {
    throw new Error("GitHub Pages requires a public HTTPS backend URL without credentials.");
  }
}

export default (phase: string): NextConfig => ({
  ...nextConfig,
  poweredByHeader: false,
  distDir: phase === PHASE_DEVELOPMENT_SERVER ? ".next-dev" : ".next",
});
