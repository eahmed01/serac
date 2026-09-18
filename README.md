# Serac

**A standalone multi-agent orchestration framework in Python.**

Serac is a Hermes-independent library for building LLM agents that reason,
call tools, share a model pool, remember across sessions, and dispatch work
to parallel workers — with the safety rails you'd want before letting an agent
touch a real machine.

It's a single package (no build step, no ORM, no service mesh) that gives you
the pieces: a robust message loop, provider abstractions, a tool registry,
SQLite-backed memory, Docker sandboxing, cost-aware model routing, structured
tracing, and retry/backoff. Use it as a library, or run the bundled CLI
entry points to get a working agent immediately.

---

## Highlights

- **Three-tier architecture** — an orchestrator decomposes work, a worker
  manager dispatches it, and workers do the actual job, all routed through a
  shared, cost-aware model pool.
- **Provider-agnostic** — local vLLM (OpenAI-compatible), Anthropic, and
  OpenAI behind one `Provider` interface. Route to the cheapest available
  capacity first.
- **Safety-first by default** — file access is workspace-restricted, terminal
  and Python execution are opt-in, and tools can run inside a network-isolated
  Docker sandbox.
- **Durable memory & sessions** — SQLite + FTS5 + vector embeddings with
  `superseded_by` semantics, plus full conversation persistence and recall.
- **Observable** — structured tracing (trace/run id propagation, JSONL spans
  for model calls, tool execution, compaction, and dispatch) and jittered
  exponential retry/backoff with error classification.
- **Tested** — 460+ unit tests across the core modules.

---

## Quick start

```bash
# 1. Point the framework at your code and run a one-shot consultation
python -m agent_framework.consult "Summarize the architecture of this project" \
    --workspace /path/to/your/project

# 2. Local model (free, fast) is the default. Pick an external model to spend:
python -m agent_framework.consult "Review the error handling" --model opus

# 3. Multi-turn: pass the same --session tag to continue the conversation
python -m agent_framework.consult "Now propose a fix" --session my-review

# 4. General-purpose dispatch with explicit tools
python -m agent_framework.dispatch \
    --tools read_file,code_search,find_files \
    "Find every place we read environment variables and list them"
```

Or use it as a library:

```python
from agent_framework import (
    AgentLoop, AgentConfig, AnthropicProvider, ToolRegistry,
)

provider = AnthropicProvider()
registry = ToolRegistry()
# registry.register(ToolDef(...))  # or register_builtin_tools(registry)

config = AgentConfig(
    provider=provider,
    system_prompt="You are a careful code reviewer.",
    tool_registry=registry,
    max_turns=25,
)
loop = AgentLoop(config)
result = loop.run("Review this diff for bugs: ...")
print(result)
```

---

## Architecture

```
                 ┌─────────────────────────────────────────────┐
   user ───────► │  Tier 1 · Orchestrator (AgentLoop)          │
                 │  message loop · steering · self-compaction  │
                 └───────────────┬─────────────────────────────┘
                                 │ decompose
                 ┌───────────────▼─────────────────────────────┐
                 │  Tier 2 · Worker Manager                    │
                 │  task decomposition · parallel dispatch     │
                 └───────────────┬─────────────────────────────┘
                                 │ route (cost-aware)
                 ┌───────────────▼─────────────────────────────┐
                 │  Model Pool                                  │
                 │  vLLM ($0) → Anthropic → OpenAI             │
                 └───────────────┬─────────────────────────────┘
                                 │
                 ┌───────────────▼─────────────────────────────┐
                 │  Tier 3 · Workers (AgentLoop instances)     │
                 │  tools · memory · sandbox · tracing         │
                 └─────────────────────────────────────────────┘
```

- **Tier 1** is the agent the user talks to. It owns the conversation loop,
  mid-turn steering, and autonomous context compaction.
- **Tier 2** takes a batch task, splits it into independent worker tasks, and
  runs them concurrently.
- **Tier 3** workers are themselves `AgentLoop` instances, each given a
  scoped toolset, the shared model pool, and optional sandboxing.

### Module map

| Module | What it does |
|---|---|
| `loop.py` | `AgentLoop` — the core message loop: model call → tool dispatch → coalesce → steer → compact. |
| `providers.py` | `Provider` ABC + `VLLMProvider`, `AnthropicProvider`, `OpenAIProvider`. |
| `tools.py` | `ToolRegistry` / `ToolDef` — JSON-schema tool definitions, role-filtered views, concurrent execution. |
| `memory.py` | SQLite memory store: CRUD, FTS5 full-text search, vector embeddings, `superseded_by`. |
| `session.py` | Persistent conversation storage — resume and search past sessions after compaction. |
| `model_pool.py` | Cost-aware routing with per-slot concurrency caps and release-on-finish. |
| `worker_manager.py` | Tier 2: task decomposition and parallel worker dispatch. |
| `sandbox.py` | Docker sandbox isolation: read-only repo, writable `/tmp`, `--network none`, non-root. |
| `llm_proxy.py` | Unix-socket proxy so sandboxed (network-none) containers can still reach LLM APIs. |
| `builtins.py` | Factory functions for built-in tools (file ops, search, git, web, terminal, todo). |
| `tool_loader.py` / `tool_resolver.py` | Config-driven tool loading; alias → fuzzy → LLM name resolution. |
| `retry.py` | `ProviderError`, error classification, jittered exponential backoff. |
| `tracing.py` | `Tracer`/`Span` with trace/run id propagation and JSONL output. |
| `tool_security.py` / `sanitize.py` | Credential scrubbing and inbound/outbound content safety boundaries. |
| `consult.py` / `dispatch.py` | CLI + programmatic entry points. |
| `price_truth*.py` | Deterministic, LLM-citable facts for price/split identity audits (opt-in add-on). |

---

## The security model

Serac is built on the assumption that an agent *will* try to read, write, and
execute things — so the default posture is restrictive:

- **Workspace-restricted reads.** File tools resolve paths against a configured
  workspace root and reject traversal outside it.
- **Execution is opt-in.** Terminal and Python execution tools are not enabled
  by default; they're added only when you explicitly opt in.
- **Docker sandboxing.** When a sandbox is configured, tools marked for
  sandbox execution run in an isolated container: read-only root, `--network
  none`, dropped capabilities, non-root, ephemeral `/tmp`. The `llm_proxy`
  provides the only outbound path, and it's logged.
- **Credential scrubbing.** Known secret shapes (API keys, tokens, host
  names) are masked in tool output before they re-enter the conversation.

Treat these as the floor, not the ceiling — layer on your own review and
approval gates for anything that ships.

---

## Configuration

Providers read credentials from the environment:

| Env var | Used by |
|---|---|
| `ANTHROPIC_API_KEY` | `AnthropicProvider` |
| `OPENAI_API_KEY` | `OpenAIProvider` |
| `EXA_API_KEY` | Exa-backed web search tool |
| `AGENT_WORKSPACE` | Default workspace root for tools |

Local vLLM defaults to `http://localhost:7999/v1` and needs no key. You can
declare named model targets in a YAML file (see `price_anomaly_model_targets.yaml`
as a reference) and resolve them by name, which keeps endpoint/model/credential
mapping out of your code.

---

## Development

```bash
# Run the test suite
python -m pytest tests/

# The framework is importable straight from a checkout (no install step)
PYTHONPATH=. python -m agent_framework.consult "hello"
```

- **No build step** — it's a plain Python package.
- **Minimal deps** — the core runs on the standard library plus the provider
  SDKs (`openai`, `anthropic`) you actually use. The ML research sandbox
  (`research_launch`, `research_sandbox.py`) is a separate, optional stack
  with its own requirements (`requirements_ml.txt`).

---

## Status

Serac is an early, actively-developed project. The core loop, providers, tools,
memory, model pool, worker manager, sandbox, retry, and tracing are in place
and tested. Expect API churn as it matures; the module map above is the
current surface.

## License

TBD — add a `LICENSE` before publishing.
