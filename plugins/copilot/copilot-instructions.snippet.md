<!--
Copy this block into your repo's .github/copilot-instructions.md so Copilot
knows the SkillOpt-Sleep tools exist. (Copilot reads copilot-instructions.md
automatically as ambient guidance.)
-->

## SkillOpt-Sleep (offline self-evolution)

This project has SkillOpt-Sleep available via an MCP server (`skillopt-sleep`).
It gives the agent a nightly "sleep cycle": it reviews past sessions, replays
recurring tasks offline, and consolidates validated memory + skills behind a
held-out gate.

When the user asks to "run the sleep cycle", "review my past sessions", "learn
my preferences", or "make the agent improve from past usage", use the MCP tools:

- `sleep_status` — what's happened + the latest staged proposal
- `sleep_dry_run` — safe preview, stages nothing
- `sleep_run` — full cycle, stages a reviewed proposal (nothing live changes)
- `sleep_adopt` — apply the staged proposal (backs up first)
- `sleep_harvest` — list mined recurring tasks
- `sleep_schedule` — install a nightly cron entry (set `hour`/`minute`)
- `sleep_unschedule` — remove the nightly cron entry

### Key parameters (pass as MCP tool arguments)

- `backend` — `mock` (default, free), `claude`, `codex`, or `copilot`
- `source` — `claude`, `codex`, or `auto` (where to read transcripts)
- `target_skill_path` — explicit SKILL.md to evolve
- `tasks_file` — pre-built TaskRecord JSON (skip harvest)
- `max_tasks` / `max_sessions` — cap workload
- `auto_adopt` — auto-adopt if the gate passes
- `json` — machine-readable output for programmatic use

### Advanced config (`~/.skillopt-sleep/config.json`)

- `preferences` — free-text house rules for the optimizer
- `gate_mode` — `on` (default) or `off`; `dream_rollouts` — >1 for more signal
- `evolve_memory` / `evolve_skill` — toggle which docs consolidate

Always show the user the held-out baseline → candidate score and the proposed
edits before suggesting `sleep_adopt`. Never hand-edit the user's memory/skill
files; only `sleep_adopt` does that, with a backup.
