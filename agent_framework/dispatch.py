#!/usr/bin/env python3
"""Dispatch Agent CLI — standalone agent with tool access.

Usage:
    # One-shot dispatch
    python -m agent_framework.dispatch_agent "Search the codebase for X and summarize findings"

    # Multi-turn with session tag
    python -m agent_framework.dispatch_agent "Continue researching X" --session my-research

    # With specific tools
    python -m agent_framework.dispatch_agent "Analyze the code" --tools read_file,code_search,terminal

    # As a Hermes consult tool (programmatic)
    from agent_framework.dispatch import dispatch_agent
    result = dispatch_agent(
        goal="Research this topic",
        session_id="my-research",
        tools=["read_file", "code_search"],
    )
    print(result.output)

Architecture:
    - Loads tools from config or defaults
    - Creates AgentLoop with model pool routing
    - Persists session state for multi-turn conversations
    - Returns structured result (output, usage, errors)
"""

from __future__ import annotations

import argparse
import json
import logging
import signal
import time
from pathlib import Path
from typing import Any, Optional

from agent_framework.loop import AgentLoop
from agent_framework.model_pool import ModelPool
from agent_framework.providers import VLLMProvider
from agent_framework.tools import ToolRegistry
from agent_framework.tool_loader import load_default_tools, ToolLoader

logger = logging.getLogger(__name__)


# Session storage path
SESSION_DIR = Path("~/.hermes/agent_sessions").expanduser()


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


def dispatch_agent(
    goal: str,
    session_id: Optional[str] = None,
    tools: Optional[list[str]] = None,
    max_turns: int = 10,
    max_tokens: int = 4096,
    vllm_url: str = "http://localhost:7999/v1",
    vllm_model: str = "Qwen/Qwen3.8-27B-FP8",
    reasoning_effort: str = "high",
    max_seconds: int | None = 900,
    system_prompt: Optional[str] = None,
) -> dict[str, Any]:
    """Dispatch an agent with tool access.

    Args:
        goal: The task/goal for the agent.
        session_id: Optional session ID for multi-turn conversations.
        tools: List of tool names to load (None = defaults).
        max_turns: Maximum conversation turns.
        max_tokens: Maximum tokens in model response.
        vllm_url: vLLM API URL.
        vllm_model: Model name.
        system_prompt: Custom system prompt.

    Returns:
        Dict with output, usage, session_id, duration, success.
    """
    start_time = time.monotonic()

    # Load session history if resuming
    history: list[str] = []
    if session_id:
        session = load_session(session_id)
        if session:
            history = session.get("history", [])

    # Build system prompt
    if system_prompt is None:
        system_prompt = (
            "You are a research agent with access to tools. "
            "Use them to gather information and complete the task. "
            "Be thorough but efficient. Return structured findings."
        )

    # Load tools
    if tools is None:
        # Default tools
        registry = load_default_tools()
    elif not tools:
        # An explicit empty list means no tools; do not load defaults.
        registry = ToolRegistry()
    else:
        # Load specific tools
        loader = ToolLoader()
        registry = loader.load(tool_names=tools)

    # Create model pool
    pool = ModelPool()
    pool.add_slot(
        VLLMProvider(base_url=vllm_url, model=vllm_model, max_tokens=max_tokens,
                     reasoning_effort=reasoning_effort),
        max_concurrent=1,
        priority=0,
        model_name="vllm",
    )

    # Route to get provider
    route = pool.route()
    provider = route.provider

    # Build agent message with history
    agent_message = goal
    if history:
        # Append previous context
        agent_message = (
            f"[Previous conversation context]\n"
            + "\n".join(history[-5:])  # Last 5 messages
            + f"\n\n[Current task]\n{goal}"
        )

    # Create and run agent loop
    loop = AgentLoop(
        provider=provider,
        system_prompt=system_prompt,
        tool_registry=registry,
        max_turns=max_turns,
        name=session_id or "dispatch",
    )

    try:
        if max_seconds is not None:
            signal.signal(signal.SIGALRM, lambda _signum, _frame: (_ for _ in ()).throw(TimeoutError(f"agent exceeded max_seconds={max_seconds}")))
            signal.alarm(max_seconds)
        output = loop.run(agent_message)
        success = bool(output.strip())
        error = "" if success else "agent loop ended without a final response"
    except Exception as e:
        output = f"Error: {e}"
        success = False
        error = str(e)
    finally:
        if max_seconds is not None:
            signal.alarm(0)
        # Release the model slot
        pool.release(route.slot_id)

    duration = time.monotonic() - start_time

    # Save session if requested
    if session_id:
        history.append(f"User: {goal}")
        history.append(f"Agent: {output}")
        save_session(session_id, {
            "history": history[-20:],  # Keep last 20 messages
            "updated_at": time.isoformat(time.localtime()),
        })

    # Build result
    usage_dict = {}
    if loop.total_usage:
        usage_dict = {
            "prompt_tokens": loop.total_usage.prompt_tokens,
            "completion_tokens": loop.total_usage.completion_tokens,
            "cost": loop.total_usage.cost,
        }

    result = {
        "output": output,
        "success": success,
        "error": error,
        "session_id": session_id,
        "duration": round(duration, 2),
        "turns_used": loop.turn_count,
        "usage": usage_dict,
    }

    return result


def main() -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Dispatch an agent with tool access",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # One-shot
  %(prog)s "Search for unusual volume features and summarize"

  # Multi-turn with session
  %(prog)s "Continue the research" --session my-research

  # With specific tools
  %(prog)s "Analyze the codebase" --tools read_file,code_search
        """,
    )
    parser.add_argument("goal", help="The task/goal for the agent")
    parser.add_argument("--session", "-s", help="Session ID for multi-turn")
    parser.add_argument("--tools", "-t", help="Comma-separated list of tools (default: all)")
    parser.add_argument("--max-turns", type=int, default=10, help="Max turns (default: 10)")
    parser.add_argument("--max-tokens", type=int, default=4096, help="Max tokens (default: 4096)")
    parser.add_argument("--vllm-url", default="http://localhost:7999/v1", help="vLLM URL")
    parser.add_argument("--vllm-model", default="Qwen/Qwen3.6-27B-FP8", help="Model name")
    parser.add_argument("--json", action="store_true", help="Output as JSON")

    args = parser.parse_args()

    # Parse tools
    tool_list = args.tools.split(",") if args.tools else None

    # Dispatch
    result = dispatch_agent(
        goal=args.goal,
        session_id=args.session,
        tools=tool_list,
        max_turns=args.max_turns,
        max_tokens=args.max_tokens,
        vllm_url=args.vllm_url,
        vllm_model=args.vllm_model,
    )

    # Output
    if args.json:
        print(json.dumps(result, indent=2))
    else:
        print(f"Agent output ({result['duration']}s, {result['turns_used']} turns):")
        print("-" * 60)
        print(result["output"])


if __name__ == "__main__":
    main()
