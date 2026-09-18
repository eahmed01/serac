"""Config-driven tool loader.

Loads tool implementations from a configuration mapping, enabling:
- Swapping tool implementations without code changes (v1 vs v2, A/B testing)
- Per-project tool selection
- Lazy loading of tool modules

Config format (YAML or dict):
    tools:
        patch: "agent_framework.builtins:patch_file"
        read_file: "agent_framework.builtins:read_file"
        # Override with custom implementation:
        # code_search: "my_project.tools:fast_code_search"
"""

from __future__ import annotations

import importlib
import logging
from typing import Any, Callable, Optional

from agent_framework.tools import ToolDef, ToolRegistry

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Tool factory registry
# ---------------------------------------------------------------------------

# Global registry of available tool implementations.
# Key: tool_name, Value: (module_path, function_name)
_AVAILABLE_TOOLS: dict[str, tuple[str, str]] = {}


def register_tool_factory(
    name: str,
    module_path: str,
    function_name: str,
) -> None:
    """Register a tool implementation in the global factory registry.

    Args:
        name: Tool name (e.g., "patch", "read_file")
        module_path: Dotted module path (e.g., "agent_framework.builtins")
        function_name: Function name that returns a ToolDef
    """
    _AVAILABLE_TOOLS[name] = (module_path, function_name)


def get_available_tools() -> dict[str, tuple[str, str]]:
    """Return a copy of the available tool implementations."""
    return dict(_AVAILABLE_TOOLS)


# ---------------------------------------------------------------------------
# Config-driven loading
# ---------------------------------------------------------------------------


def load_tools_from_config(
    config: dict[str, str],
    sandbox: Optional[Any] = None,
    reject_host_only: bool = False,
    reject_host_only_names: Optional[set[str]] = None,
    reject_requires_sandbox_names: Optional[set[str]] = None,
) -> ToolRegistry:
    """Load tools from a configuration mapping.

    Args:
        config: Dict mapping tool names to implementation spec.
            Values can be:
            - "module:function" — full spec
            - Just a name — looks up in the global factory registry
        sandbox: Optional Sandbox instance passed to tool factories.
        reject_host_only: Raise instead of skipping a host-only tool when a
            sandbox is provided. Used for explicitly requested tools.
        reject_host_only_names: Names that were explicitly requested and must
            not be silently filtered.
        reject_requires_sandbox_names: Names that were explicitly requested
            without a sandbox and must raise instead of being omitted. If
            omitted, every config entry is treated as explicit.

    Returns:
        ToolRegistry with all loaded tools.

    Example:
        config = {
            "patch": "agent_framework.builtins:patch_file",
            "read_file": "agent_framework.builtins:read_file",
            "math": "my_project.tools:my_math_tool",
        }
        registry = load_tools_from_config(config)
    """
    registry = ToolRegistry()

    for tool_name, impl_spec in config.items():
        try:
            tool_def = _load_tool_impl(tool_name, impl_spec, sandbox)
            if sandbox is not None and tool_def.execution_mode != "sandbox":
                message = (
                    f"Tool '{tool_name}' is host-only and cannot be loaded "
                    "when a sandbox is configured"
                )
                if reject_host_only or (
                    reject_host_only_names is not None and tool_name in reject_host_only_names
                ):
                    raise ValueError(message)
                logger.info("Skipping %s", message)
                continue
            if sandbox is None and tool_def.requires_sandbox:
                message = (
                    f"Tool '{tool_name}' requires a sandbox and cannot be loaded "
                    "without one"
                )
                explicit_names = (
                    set(config)
                    if reject_requires_sandbox_names is None
                    else reject_requires_sandbox_names
                )
                if tool_name in explicit_names:
                    raise ValueError(message)
                logger.info("Skipping %s", message)
                continue
            registry.register(tool_def)
        except ValueError as exc:
            if (
                (reject_host_only or (reject_host_only_names is not None and tool_name in reject_host_only_names))
                and "host-only" in str(exc)
            ):
                raise
            if "requires a sandbox" in str(exc):
                explicit_names = (
                    set(config)
                    if reject_requires_sandbox_names is None
                    else reject_requires_sandbox_names
                )
                if tool_name in explicit_names:
                    raise
            logger.error("Failed to load tool '%s' (%s): %s", tool_name, impl_spec, exc)
        except Exception as exc:
            logger.error("Failed to load tool '%s' (%s): %s", tool_name, impl_spec, exc)

    return registry


def _load_tool_impl(
    tool_name: str,
    impl_spec: str,
    sandbox: Optional[Any] = None,
) -> ToolDef:
    """Load a single tool implementation.

    Args:
        tool_name: Logical tool name (for error messages).
        impl_spec: Either "module:function" or a registry name.
        sandbox: Optional Sandbox instance.

    Returns:
        ToolDef instance.
    """
    # Check if it's a colon-separated module:function spec
    if ":" in impl_spec:
        module_path, function_name = impl_spec.rsplit(":", 1)
    elif impl_spec in _AVAILABLE_TOOLS:
        module_path, function_name = _AVAILABLE_TOOLS[impl_spec]
    else:
        raise ValueError(
            f"Tool '{tool_name}': '{impl_spec}' is not a valid spec "
            f"(expected 'module:function' or a registered name)"
        )

    # Dynamic import
    module = importlib.import_module(module_path)
    factory = getattr(module, function_name)

    # Call factory with optional sandbox
    if callable(factory):
        import inspect
        sig = inspect.signature(factory)
        if "sandbox" in sig.parameters:
            tool_def = factory(sandbox=sandbox)
        else:
            tool_def = factory()
    else:
        # Direct ToolDef (no factory function)
        tool_def = factory

    if not isinstance(tool_def, ToolDef):
        raise TypeError(
            f"Tool '{tool_name}': factory returned {type(tool_def)}, expected ToolDef"
        )

    return tool_def


# ---------------------------------------------------------------------------
# Default configs
# ---------------------------------------------------------------------------


def default_file_tools(sandbox: Optional[Any] = None) -> dict[str, str]:
    """Default file operation tools config."""
    return {
        "read_file": "agent_framework.builtins:read_file_factory",
        "write_file": "agent_framework.builtins:write_file_factory",
        "patch_file": "agent_framework.builtins:patch_file_factory",
    }


def default_search_tools(sandbox: Optional[Any] = None) -> dict[str, str]:
    """Default search tools config."""
    return {
        "code_search": "agent_framework.builtins:code_search_factory",
        "find_files": "agent_framework.builtins:find_files_factory",
        "web_search": "agent_framework.builtins:web_search_factory",
        "sec_search": "agent_framework.builtins:sec_search_factory",
        "sec_fetch": "agent_framework.builtins:sec_fetch_factory",
    }


def default_exec_tools(sandbox: Optional[Any] = None) -> dict[str, str]:
    """Default execution tools config."""
    return {
        "execute_terminal": "agent_framework.builtins:execute_terminal_factory",
        "execute_python": "agent_framework.builtins:execute_python_factory",
    }


def default_research_tools(sandbox: Optional[Any] = None) -> dict[str, str]:
    """Research sandbox tools (persistent Python namespace via the bundled sandbox server)."""
    return {
        "research_execute": "agent_framework.builtins:research_execute_factory",
    }


def default_git_tools(sandbox: Optional[Any] = None) -> dict[str, str]:
    """Default git tools config."""
    return {
        "git_status": "agent_framework.builtins:git_status_factory",
        "git_diff": "agent_framework.builtins:git_diff_factory",
        "git_log": "agent_framework.builtins:git_log_factory",
    }


def default_all_tools(sandbox: Optional[Any] = None) -> dict[str, str]:
    """Default config for all built-in tools."""
    all_configs = [
        default_file_tools(sandbox),
        default_search_tools(sandbox),
        default_exec_tools(sandbox),
        default_git_tools(sandbox),
        default_research_tools(sandbox),
        {
            "todo": "agent_framework.builtins:todo_factory",
        },
    ]
    merged: dict[str, str] = {}
    for cfg in all_configs:
        merged.update(cfg)
    return merged


# ---------------------------------------------------------------------------
# Convenience: load all defaults
# ---------------------------------------------------------------------------


def load_default_tools(sandbox: Optional[Any] = None) -> ToolRegistry:
    """Load all default tools into a ToolRegistry.

    Args:
        sandbox: Optional Sandbox instance.

    Returns:
        ToolRegistry with all default tools.
    """
    config = default_all_tools(sandbox)
    return load_tools_from_config(
        config,
        sandbox=sandbox,
        reject_requires_sandbox_names=set(),
    )


# ---------------------------------------------------------------------------
# ToolLoader: reusable, composable loader
# ---------------------------------------------------------------------------


class ToolLoader:
    """Reusable, composable tool loader.

    Enables:
    - Loading tools from config at runtime
    - Overriding specific tool implementations
    - A/B testing different tool versions
    - Per-project tool selection

    Example:
        loader = ToolLoader(sandbox=my_sandbox)
        registry = loader.load()  # default tools

        # Override one tool:
        registry = loader.load(overrides={
            "patch_file": "my_project.tools:patch_v2_factory",
        })

        # Load only specific tools:
        registry = loader.load(tool_names=["read_file", "code_search"])
    """

    def __init__(self, sandbox: Optional[Any] = None) -> None:
        self.sandbox = sandbox

    def load(
        self,
        tool_names: Optional[list[str]] = None,
        overrides: Optional[dict[str, str]] = None,
        reject_host_only_names: Optional[set[str]] = None,
    ) -> ToolRegistry:
        """Load tools.

        Args:
            tool_names: If provided, load only these tools. Otherwise load all defaults.
            overrides: Dict mapping tool names to custom implementation specs.
            reject_host_only_names: Names that must not be silently filtered when a
                sandbox is configured. By default, all names in ``tool_names`` and
                ``overrides`` are treated as explicit. Pass an empty set when a
                caller is selecting a safe default list and wants host-only tools
                filtered rather than rejected.

        Returns:
            ToolRegistry with loaded tools.
        """
        config = default_all_tools(self.sandbox)

        # Filter to specific tools if requested
        if tool_names is not None:
            config = {k: v for k, v in config.items() if k in tool_names}

        # Apply overrides
        if overrides:
            config.update(overrides)

        explicit_names = (
            set(tool_names or ()) | set((overrides or {}).keys())
            if reject_host_only_names is None
            else reject_host_only_names
        )
        return load_tools_from_config(
            config,
            sandbox=self.sandbox,
            reject_host_only_names=explicit_names if self.sandbox is not None else None,
            reject_requires_sandbox_names=explicit_names,
        )
