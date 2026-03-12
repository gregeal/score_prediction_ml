import type { NextConfig } from "next";

const repoName = process.env.GITHUB_REPOSITORY?.split("/")[1] || "score_prediction_ml";
const isGithubPagesBuild = process.env.GITHUB_PAGES === "true";
const basePath = isGithubPagesBuild ? `/${repoName}` : "";

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

export default nextConfig;
