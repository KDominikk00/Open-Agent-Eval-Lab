.PHONY: install test lint typecheck check baselines demo clean

install:
	pip install -e ".[dev]"

test:
	pytest --cov

lint:
	ruff check .

typecheck:
	mypy

check: lint typecheck test

# Regenerate the committed offline baselines and the local leaderboard site.
baselines:
	scripts/publish_baselines.sh

# Run one agent end to end and open the report.
demo:
	rm -rf results/demo
	open-agent-eval run --tasks-dir tasks --output-dir results --agent keyword-retriever \
		--split public --label smoke --run-id demo
	open-agent-eval leaderboard --results-dir results --output-dir leaderboard --include-smoke
	@echo "Open leaderboard/leaderboard.html"

clean:
	rm -rf results-ci leaderboard-ci benchmarks/site .pytest_cache .mypy_cache .ruff_cache .coverage
