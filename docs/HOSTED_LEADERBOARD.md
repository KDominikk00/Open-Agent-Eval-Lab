# Hosted Leaderboard Deployment

This repository includes an automated GitHub Pages deployment workflow:

- Workflow file: `.github/workflows/deploy-leaderboard.yml`
- Triggered on pushes to `main` and manual `workflow_dispatch`

## What the workflow does

1. Validates benchmark tasks.
2. Builds the leaderboard and trace viewer from the committed baselines in `benchmarks/published/` (regenerate them with `make baselines`).
3. Publishes the static site to GitHub Pages.

The site is built from committed results rather than fresh runs, so what is published is exactly what was reviewed in the repository.

## Enable in repository settings

1. Open **Settings** in your GitHub repository.
2. Open **Pages**.
3. Set **Build and deployment** source to **GitHub Actions**.

After that, each successful workflow run publishes the latest leaderboard.
