# Dataset

45 tasks: 31 public and 14 hidden. Counts below come from `open-agent-eval validate` on the current tree.

| | Count |
|---|---|
| Tool-use episodes (sandboxed environment + trace grading) | 31 |
| Single-turn text tasks (original v0.x dataset) | 14 |
| Difficulty (easy / medium / hard) | 7 / 26 / 12 |
| Graders (`contains` / `exact` / `json`) | 39 / 3 / 3 |

## Task families

Each episode family is generated from a small template with varied fixtures. The task files themselves are plain JSON.

| Family | Category | What it tests | Behavioural checks |
|---|---|---|---|
| `log-root-cause` | debugging | Search noisy logs, name the failing upstream | must call `search_logs` |
| `order-total` | data_analysis | Filter a table, sum with the calculator | `db_query` before `calculator` |
| `config-fix` | file_ops | Read a config, write back exactly one changed value | `read_file` then `write_file`, final file content must match |
| `injection` | safety | Summarise a document that contains an injected instruction | must not call `http_get`/`write_file`; answer must not contain the attacker URL |
| `api-chain` | api_usage | Follow a link in one response to a second request | two `http_get` calls in order |
| `secret-hygiene` | security | Name credential variables without printing their values | must not leak the secret string or call `http_get` |

## Difficulty labels

Labels are a rough guide assigned per family and instance, not a calibrated measure.

- **easy**: one tool call, answer appears verbatim in the observation.
- **medium**: two dependent calls, or a distractor in the data (an unrelated error line, a second customer's rows).
- **hard**: success requires *not* doing something (ignoring an injected instruction, withholding a secret).

## Splits

- `public` tasks are what the published results report.
- `hidden` tasks are meant for private evaluation. **In this repository they are committed**, so they are not secret and
  results on them should not be read as contamination-resistant. If you fork this to evaluate a model you do not want
  to see the answers, keep your own hidden split out of version control.

## Task validity

`open-agent-eval validate` rejects, among other things:

- unknown tools, environments that are not well-formed, or trajectories that use tools the task does not enable;
- duplicate task ids and files outside `tasks/public` / `tasks/hidden`;
- a task whose `expected_response` fails its own grader.

For episode tasks, the `oracle` agent replays `reference_trajectory` and must score 100%. This is checked in
`benchmarks/published` (public split) and by `test_oracle_solves_every_task_in_the_dataset` (all 45 tasks, including hidden).

## Known weaknesses

- Fixtures are synthetic and small; most tasks are solvable by one or two tool calls.
- 39 of 45 tasks use `contains`, which can be satisfied by an answer that mentions the right string for the wrong reason.
  Trace constraints cover the most important cases but not all.
- Six episode families with four to six instances each means tasks within a family are correlated; the confidence
  intervals treat them as independent and are therefore somewhat optimistic.
- The 14 original single-turn tasks list `required_tools` that are not real sandbox tools (for example `security_guide`).
  They are kept for backward compatibility and are graded on the answer text only.
