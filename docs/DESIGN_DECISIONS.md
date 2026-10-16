# Design decisions and limitations

## Decisions

**Simulated tools, not real ones.** Tools run against an in-memory fixture. Runs are deterministic, offline, free,
and cannot damage anything, so results are comparable across machines and CI can verify them. The cost is realism:
a model that handles clean fixtures may still fail against real logs or flaky APIs.

**Grade behaviour, not only the answer.** A correct answer reached by calling a forbidden tool, or after following an
injected instruction, is a failure. Without trace grading, most safety tasks reduce to string matching.

**Binary scoring.** A task attempt scores 0 or 1. This keeps the pass rate a plain proportion so Wilson intervals
apply. The average score therefore equals the pass rate for current graders; it exists for future partial-credit graders.

**Tasks are the independent unit for confidence intervals.** With `--trials N`, results are averaged per task first, so
extra trials reduce sampling noise from the agent but do not make the dataset look larger.

**Every task must be solvable by construction.** A `reference_trajectory` plus the `oracle` agent proves each episode
task has a path to a pass under its own constraints. Validation also rejects tasks whose reference answer fails
their own grader. This caught a real bug in the original dataset during this rebuild (`public.security.parameterized-query`).

**Zero runtime dependencies.** HTTP uses `urllib`, statistics use the standard library, the leaderboard is static HTML.
Installing the framework never conflicts with an agent's own dependencies.

**Baselines that bracket the range.** `oracle` is the ceiling, `random` the floor, `keyword-retriever` a cheap
heuristic. `echo-reference` returns the reference answer without using any tool: it scores 26% on the public split,
which shows how much of the dataset can be passed without doing the work (see limitations).

## Limitations

- **No real-model results are published.** The committed leaderboard contains only offline reference agents. The
  OpenAI, Anthropic and Ollama adapters are tested against mocked HTTP, not live servers. Publishing real results is open work.
- **The hidden split is not hidden.** It lives in the public repository. See `docs/DATASET.md`.
- **Small, templated dataset.** 31 public tasks from six families. Intervals are wide (the oracle's 100% still has a
  lower bound of 89%) and, because tasks within a family are correlated, somewhat optimistic.
- **Answer matching is lenient.** `contains` can pass an answer that includes the right string among wrong ones.
- **Single-run determinism covers outcomes, not timing.** Latency in the offline baselines measures Python overhead
  and says nothing about model speed.
- **Provider adapters send one tool call per turn** (`parallel_tool_calls` disabled for OpenAI; the first `tool_use`
  block is used for Anthropic), so parallel tool use is not evaluated.

## Possible next steps

Run the Ollama profile on a real machine and publish the results; partial-credit graders; larger
and less templated task families; per-tool failure analysis.
