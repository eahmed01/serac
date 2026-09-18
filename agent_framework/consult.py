#!/usr/bin/env python3
"""Consultant Agent — multi-turn agent with model targeting and workspace-safe tools.

A secure, model-agnostic agent dispatcher for research and consultation tasks.
Supports local vLLM and external models (Anthropic, OpenAI) with a minimal,
workspace-restricted toolset.

Security model:
- File reads restricted to configured workspace root (no path traversal)
- No terminal/Python execution by default (opt-in only)
- Web search available for initial research
- External models (Anthropic/OpenAI) require explicit --model selection
- Tool-by-tool opt-in via --tools flag

Usage:
    # Local vLLM (fast, free)
    python -m agent_framework.consult "Summarize the agent framework architecture"

    # External model (costs money)
    python -m agent_framework.consult "Review this code" --model opus

    # Multi-turn session
    python -m agent_framework.consult "Continue the analysis" --session project-review

    # With workspace restriction
    python -m agent_framework.consult "Find security issues" --workspace ~/dev/project/0

    # Programmatic (Hermes integration)
    from agent_framework.consult import consult
    result = consult(
        goal="Research this topic",
        model="opus",  # or "local", "sonnet", etc.
        session_id="my-research",
        workspace="/home/user/dev/project/0",
    )

Available models:
    local     — vLLM on localhost:7999 (default)
    opus      — Anthropic Claude Opus 4.8
    sonnet    — Anthropic Claude Sonnet 5
    fable     — Anthropic Claude Fable 5
    gpt55     — OpenAI GPT-5.5
    gpt41     — OpenAI GPT-4.1

Default tools (workspace-safe):
    - read_file   (workspace-restricted)
    - find_files  (workspace-restricted)
    - code_search (workspace-restricted)
    - web_search  (for initial research)
    - sec_search  (SEC filing search)
    - sec_fetch   (SEC filing retrieval)

Optional tools (explicit --tools required):
    - execute_terminal
    - execute_python
    - write_file   (dangerous, opt-in only)
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

# Load API keys from ~/.exa.api (Exa) and .env files
_ENV_PATHS = [
    Path.home() / ".exa.api",  # Exa API key
    Path.home() / ".hermes" / "profiles" / "default" / ".env",
]
for _env_path in _ENV_PATHS:
    if _env_path.exists():
        for line in _env_path.read_text().splitlines():
            if line and not line.startswith("#") and "=" in line:
                key, _, value = line.partition("=")
                os.environ.setdefault(key.strip(), value.strip())

from agent_framework.loop import AgentLoop
from agent_framework.model_pool import ModelPool
from agent_framework.providers import AnthropicProvider, OpenAIProvider, VLLMProvider
from agent_framework.sanitize import CONSULTANT_WEB_CONTENT_INSTRUCTION
from agent_framework.tool_loader import ToolLoader
from agent_framework.tools import ToolDef, ToolRegistry

logger = logging.getLogger(__name__)

# Session storage
SESSION_DIR = Path.home() / ".hermes" / "agent_sessions"

# Default workspace root (repo root)
DEFAULT_WORKSPACE = Path.home() / "dev" / "project" / "0"

# Available models — same as consultation_ask
MODEL_CONFIGS = {
    "local": {
        "provider": "vllm",
        "url": "http://localhost:7999/v1",
        "model": "Qwen/Qwen3.8-27B-FP8",
        "max_tokens": 65536,
        "reasoning_effort": "high",
    },
    "opus": {
        "provider": "anthropic",
        "model": "claude-opus-4-8",
        "max_tokens": 8192,
    },
    "sonnet": {
        "provider": "anthropic",
        "model": "claude-sonnet-5",
        "max_tokens": 8192,
    },
    "fable": {
        "provider": "anthropic",
        "model": "claude-fable-5",
        "max_tokens": 8192,
    },
    "gpt55": {
        "provider": "openai",
        "model": "gpt-5.5",
        "max_tokens": 8192,
    },
    "gpt41": {
        "provider": "openai",
        "model": "gpt-4.1",
        "max_tokens": 8192,
    },
}

# Safe default tools (workspace-restricted, read-only)
SAFE_TOOLS = ["read_file", "find_files", "code_search", "web_search", "sec_search", "sec_fetch"]

# System prompt template
CONSULTANT_PROMPT = """\
You are a research consultant with access to tools for gathering information.

WORKSPACE: {workspace}
You may ONLY read files within the workspace directory above.
Do NOT access files outside the workspace.

TOOLS AVAILABLE: {tools_list}

{sandbox_instructions}
{todo_instruction}
GUIDELINES:
- Use web_search for initial research or finding public information
- Use read_file, find_files, code_search to explore the workspace
- Be thorough but efficient — avoid unnecessary tool calls
- Return structured, actionable findings
- If a file is outside the workspace, refuse to read it
- When in doubt, ask for clarification before proceeding
- TERMINATE: After collecting information, STOP calling tools and provide your analysis
- Do NOT loop calling the same tool repeatedly
- If you have enough information to answer the question, end the conversation
{web_content_instruction}
"""

# Instructions appended when running with a Docker sandbox
SANDBOX_INSTRUCTIONS = """\
SANDBOX MODE: You are running in a Docker container.
- The project repository is mounted read-only at /repo
- Use relative paths (e.g., "agent_framework/sandbox.py") or /repo/ paths
- Write operations go to /tmp/workspace (writable)
- No network access is available from tools
"""


# ---------------------------------------------------------------------------
# Session management
# ---------------------------------------------------------------------------

def load_session(session_id: str) -> Optional[dict[str, Any]]:
    """Load session state from disk."""
    session_file = SESSION_DIR / f"{session_id}.json"
    if session_file.exists():
        with open(session_file) as f:
            return json.load(f)
    return None


def save_session(session_id: str, state: dict[str, Any]) -> None:
    """Save session state to disk."""
    SESSION_DIR.mkdir(parents=True, exist_ok=True)
    session_file = SESSION_DIR / f"{session_id}.json"
    with open(session_file, "w") as f:
        json.dump(state, f, indent=2)


# ---------------------------------------------------------------------------
# Provider factory
# ---------------------------------------------------------------------------

def create_provider(model: str, max_tokens: Optional[int] = None):
    """Create a provider for the given model name.

    Args:
        model: Model name from MODEL_CONFIGS or custom provider spec.
        max_tokens: Override max tokens.

    Returns:
        Provider instance.
    """
    config = MODEL_CONFIGS.get(model)
    if not config:
        raise ValueError(
            f"Unknown model: {model}. Available: {', '.join(MODEL_CONFIGS.keys())}"
        )

    effective_tokens = max_tokens or config["max_tokens"]

    if config["provider"] == "vllm":
        return VLLMProvider(
            base_url=config["url"],
            model=config["model"],
            max_tokens=effective_tokens,
            reasoning_effort=config.get("reasoning_effort", "high"),
        )
    elif config["provider"] == "anthropic":
        api_key = os.environ.get("ANTHROPIC_API_KEY")
        if not api_key:
            raise ValueError(
                "Anthropic API key not set. Set ANTHROPIC_API_KEY environment variable."
            )
        return AnthropicProvider(
            api_key=api_key,
            model=config["model"],
            max_tokens=effective_tokens,
        )
    elif config["provider"] == "openai":
        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            raise ValueError(
                "OpenAI API key not set. Set OPENAI_API_KEY environment variable."
            )
        return OpenAIProvider(
            api_key=api_key,
            model=config["model"],
            max_tokens=effective_tokens,
        )
    else:
        raise ValueError(f"Unknown provider type: {config['provider']}")


# ---------------------------------------------------------------------------
# Workspace-safe tools
# ---------------------------------------------------------------------------

def load_workspace_tools(
    workspace: str | Path,
    tool_names: Optional[list[str]] = None,
    sandbox: Any = None,
) -> ToolRegistry:
    """Load tools with workspace restrictions.

    Args:
        workspace: Root directory for file operations.
        tool_names: Explicit tool list (None = safe defaults).
        sandbox: Optional Sandbox instance. When provided, tools execute
            inside Docker (sandbox handles all security; no path wrapper needed).

    Returns:
        ToolRegistry with workspace-restricted tools.
    """
    workspace_path = Path(workspace).resolve()

    loader = ToolLoader(sandbox=sandbox)
    if tool_names is None:
        # SAFE_TOOLS is the consultant default in both modes. With a sandbox,
        # host-only entries are filtered rather than treated as explicit asks.
        registry = loader.load(
            tool_names=SAFE_TOOLS,
            reject_host_only_names=set() if sandbox is not None else None,
        )
        loaded_tool_names = set(registry.tools)
    elif not tool_names:
        # An explicitly empty list is authoritative: load no tools.
        registry = ToolRegistry()
        loaded_tool_names = set()
    else:
        registry = loader.load(tool_names=tool_names)
        loaded_tool_names = set(tool_names)

    # Host-side path tools need an explicit boundary. A sandbox already
    # provides that boundary and its executors use container paths instead.
    if sandbox is None:
        for tool_name in loaded_tool_names:
            restrict_tool_to_workspace(registry, tool_name, workspace_path)

    return registry


def restrict_tool_to_workspace(
    registry: ToolRegistry,
    tool_name: str,
    workspace: Path,
) -> None:
    """Wrap a path-based executor with a resolved workspace boundary.

    Relative paths are interpreted relative to ``workspace`` (rather than
    the process CWD), and the resolved path is passed to the underlying
    executor. Resolving before the containment check also rejects symlink
    escapes.
    """
    tools = registry._tools
    if tool_name not in tools:
        return

    original_def = tools[tool_name]
    original_executor = original_def.executor

    # Determine which parameter name this tool uses for file path
    # Read the parameters schema to find the path parameter
    path_param = None
    props = original_def.parameters.get("properties", {})
    for param_name in ["path", "file_path", "filename"]:
        if param_name in props:
            path_param = param_name
            break

    if path_param is None:
        # No path parameter found, skip restriction
        return

    def restricted_executor(**kwargs: Any) -> str:
        file_path = kwargs.get(path_param)
        if file_path is None:
            if path_param in original_def.parameters.get("required", []):
                return f"ERROR: {path_param} is required"
            # Optional path parameters default to the workspace root, matching
            # the tools' ``."`` defaults without consulting process CWD.
            file_path = "."

        # Resolve relative paths from the configured workspace. ``resolve``
        # also follows symlinks, so symlink escapes are rejected below.
        requested = Path(file_path)
        target = (workspace / requested if not requested.is_absolute() else requested).resolve()

        # Check if it's within workspace
        try:
            target.relative_to(workspace)
        except ValueError:
            return (
                f"ERROR: Path '{file_path}' is outside the workspace. "
                f"Allowed root: {workspace}"
            )

        # Pass the resolved path onward; otherwise a relative path would be
        # interpreted against the host process CWD by host executors.
        kwargs[path_param] = str(target)
        return original_executor(**kwargs)
    tools[tool_name] = ToolDef(
        name=original_def.name,
        description=original_def.description,
        parameters=original_def.parameters,
        executor=restricted_executor,
        role=original_def.role,
        execution_mode=original_def.execution_mode,
        requires_sandbox=original_def.requires_sandbox,
    )


# ---------------------------------------------------------------------------
# Consult function (programmatic API)
# ---------------------------------------------------------------------------

def consult(
    goal: str,
    model: str = "local",
    session_id: Optional[str] = None,
    workspace: Optional[str | Path] = None,
    tools: Optional[list[str]] = None,
    max_turns: int = 10,
    max_tokens: Optional[int] = None,
    system_prompt: Optional[str] = None,
    attach_files: Optional[list[str]] = None,
    sandbox: Optional[Any] = None,
) -> dict[str, Any]:
    """Run a consultant agent.

    Args:
        goal: The research task or question.
        model: Model name from MODEL_CONFIGS (default: local vLLM).
        session_id: Session ID for multi-turn conversations.
        workspace: Workspace root for file operations.
        tools: Explicit tool list (None = safe defaults).
        max_turns: Maximum conversation turns.
        max_tokens: Override model max tokens.
        system_prompt: Custom system prompt (overrides default).
        attach_files: List of file paths to read and inject into the prompt
            before the goal. Files are truncated at 50K chars each.
            The consultant sees them inline — no tool calls needed.
        sandbox: Optional Sandbox instance. When provided, file and code
            execution tools run inside Docker containers for isolation.

    Returns:
        Dict with output, usage, duration, turns, success/error.
    """
    start_time = time.monotonic()

    # Resolve workspace
    workspace_path = Path(workspace or DEFAULT_WORKSPACE).expanduser().resolve()

    # Determine the workspace path to show the agent
    # When running in a sandbox, the agent sees /repo (container path)
    # When running without sandbox, it sees the actual workspace path
    agent_workspace = "/repo" if sandbox is not None else workspace_path

    # Load session history if resuming
    history: list[str] = []
    if session_id:
        session = load_session(session_id)
        if session:
            history = session.get("history", [])

    # Load tools before building the prompt so the advertised tool list matches
    # the registry after sandbox filtering.
    registry = load_workspace_tools(workspace_path, tools, sandbox=sandbox)

    # Build system prompt
    tools_list = list(registry.tools)
    # Include web content security instruction only when web_search is available
    has_web = "web_search" in tools_list
    web_instruction = CONSULTANT_WEB_CONTENT_INSTRUCTION if has_web else ""
    # Include sandbox instructions only when running with Docker sandbox
    sandbox_instruction = SANDBOX_INSTRUCTIONS if sandbox is not None else ""
    # Include todo instructions only when todo tool is available
    has_todo = "todo" in tools_list
    from agent_framework.todo import TODO_SYSTEM_INSTRUCTION
    todo_instruction = TODO_SYSTEM_INSTRUCTION if has_todo else ""
    if system_prompt is None:
        system_prompt = CONSULTANT_PROMPT.format(
            workspace=agent_workspace,
            tools_list=", ".join(tools_list),
            sandbox_instructions=sandbox_instruction,
            todo_instruction=todo_instruction,
            web_content_instruction=web_instruction,
        )

    # Create provider
    provider = create_provider(model, max_tokens)

    # Read attached files
    attached_blocks: list[str] = []
    if attach_files:
        for fp in attach_files:
            # Attachments are host-side reads even when the agent tools run in
            # Docker. Resolve from the configured workspace (not CWD), follow
            # symlinks, and reject anything outside that workspace before any
            # read is attempted. Never put the caller's spelling or host path
            # into model-visible content.
            requested = Path(fp).expanduser()
            target = (
                workspace_path / requested
                if not requested.is_absolute()
                else requested
            ).resolve()
            try:
                relative = target.relative_to(workspace_path)
            except ValueError:
                attached_blocks.append(
                    "=== ERROR READING ATTACHED FILE: outside workspace ==="
                )
                continue

            relative_label = relative.as_posix()
            if relative_label == ".":
                relative_label = ""
            label = (
                "/repo" + (f"/{relative_label}" if relative_label else "")
                if sandbox is not None
                else relative_label or "."
            )
            try:
                content = target.read_text(errors="replace")
                if len(content) > 50_000:
                    content = content[:50_000] + "\n... [truncated, >50K chars]"
                attached_blocks.append(
                    f"=== ATTACHED FILE: {label} ===\n"
                    f"{content}\n=== END: {label} ==="
                )
            except FileNotFoundError:
                attached_blocks.append(f"=== MISSING FILE: {label} ===")
            except Exception as e:
                # Do not interpolate the exception: some OS errors include
                # the host path. Keep the existing error-block semantics while
                # preventing disclosure through diagnostics.
                attached_blocks.append(
                    f"=== ERROR READING: {label} → {type(e).__name__} ==="
                )

    # Build agent message with attachments + history
    agent_message = goal
    if attached_blocks:
        agent_message = "\n\n".join(attached_blocks) + f"\n\n[Current task]\n{goal}"
    if history:
        agent_message = (
            "[Previous conversation context]\n"
            + "\n".join(history[-5:])  # Last 5 exchanges
            + f"\n\n{agent_message}"
        )

    # Create and run agent loop
    loop = AgentLoop(
        provider=provider,
        system_prompt=system_prompt,
        tool_registry=registry,
        max_turns=max_turns,
        name=session_id or "consult",
    )

    try:
        output = loop.run(agent_message)
        success = True
        error = ""
    except Exception as e:
        output = f"Error: {e}"
        success = False
        error = str(e)

    duration = time.monotonic() - start_time

    # Save session if requested
    if session_id:
        history.append(f"User: {goal}")
        history.append(f"Agent: {output[:2000]}")  # Truncate for storage
        save_session(session_id, {
            "history": history[-20:],  # Keep last 20 messages
            "model": model,
            "updated_at": datetime.now().isoformat(),
            "workspace": str(workspace_path),
        })

    # Build result
    usage_dict = {}
    if loop.total_usage:
        usage_dict = {
            "prompt_tokens": loop.total_usage.prompt_tokens,
            "completion_tokens": loop.total_usage.completion_tokens,
            "cost": loop.total_usage.cost,
        }

    return {
        "output": output,
        "success": success,
        "error": error,
        "session_id": session_id,
        "model": model,
        "duration": round(duration, 2),
        "turns_used": loop.turn_count,
        "usage": usage_dict,
        "workspace": str(workspace_path),
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Consultant Agent — multi-turn research with model targeting",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Models:
  local     vLLM (Qwen3.6-27B, localhost:7999, free)
  opus      Claude Opus 4.8 (Anthropic, costs $)
  sonnet    Claude Sonnet 5 (Anthropic, costs $)
  fable     Claude Fable 5 (Anthropic, costs $)
  gpt55     GPT-5.5 (OpenAI, costs $)
  gpt41     GPT-4.1 (OpenAI, costs $)

Tools (default: safe read-only):
  read_file, find_files, code_search, web_search

Optional (add with --tools):
  execute_terminal, execute_python, write_file

Examples:
  # Local model, default tools
  %(prog)s "Summarize the agent framework architecture"

  # External model
  %(prog)s "Review this code" --model opus

  # Multi-turn session
  %(prog)s "Continue the analysis" --session project-review

  # With specific tools
  %(prog)s "Analyze the codebase" --tools read_file,code_search,terminal
        """,
    )
    parser.add_argument("goal", help="Research task or question")
    parser.add_argument(
        "--model", "-m",
        default="local",
        choices=list(MODEL_CONFIGS.keys()),
        help="Model to use (default: local)",
    )
    parser.add_argument(
        "--session", "-s",
        help="Session ID for multi-turn conversations",
    )
    parser.add_argument(
        "--workspace", "-w",
        default=str(DEFAULT_WORKSPACE),
        help="Workspace root for file operations",
    )
    parser.add_argument(
        "--tools", "-t",
        help="Comma-separated tool list (default: safe tools)",
    )
    parser.add_argument(
        "--max-turns", type=int, default=10,
        help="Maximum conversation turns (default: 10)",
    )
    parser.add_argument(
        "--max-tokens", type=int,
        help="Override model max tokens",
    )
    parser.add_argument(
        "--json", action="store_true",
        help="Output as JSON",
    )

    args = parser.parse_args()

    # Parse tools
    tool_list = args.tools.split(",") if args.tools else None

    # Dispatch
    result = consult(
        goal=args.goal,
        model=args.model,
        session_id=args.session,
        workspace=args.workspace,
        tools=tool_list,
        max_turns=args.max_turns,
        max_tokens=args.max_tokens,
    )

    # Output
    if args.json:
        print(json.dumps(result, indent=2))
    else:
        model_label = f" ({args.model})" if args.model != "local" else ""
        print(
            f"Consultant output{model_label} "
            f"({result['duration']}s, {result['turns_used']} turns):"
        )
        print("-" * 60)
        print(result["output"])
        if result["usage"]:
            usage = result["usage"]
            print("-" * 60)
            print(
                f"Usage: {usage['prompt_tokens']} prompt, "
                f"{usage['completion_tokens']} completion, "
                f"${usage['cost']:.4f}"
            )


if __name__ == "__main__":
    main()
