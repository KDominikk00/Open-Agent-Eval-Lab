# Hosted Leaderboard Deployment

This repository includes an automated GitHub Pages deployment workflow:

- Workflow file: `.github/workflows/deploy-leaderboard.yml`
- Triggered on pushes to `main` and manual `workflow_dispatch`

## What the workflow does

1. Validates benchmark tasks.
2. Runs two baseline benchmark passes.
3. Generates leaderboard artifacts.
4. Publishes the generated static site to GitHub Pages.

## Enable in repository settings

1. Open **Settings** in your GitHub repository.
2. Open **Pages**.
3. Set **Build and deployment** source to **GitHub Actions**.

After that, each successful workflow run publishes the latest leaderboard.
