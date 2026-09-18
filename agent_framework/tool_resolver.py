"""Tool name resolver: maps model-requested tool names to registered names.

Models often call tools by slightly different names than what's registered.
This module provides a pluggable resolution pipeline:

  1. Alias lookup (fast, config-driven)
  2. Fuzzy match (medium, similarity-based)
  3. LLM policy (slow, for hard cases)

Usage:
    resolver = ToolResolver(
        aliases={"file_read": "read_file"},
        registry=my_registry,
    )
    canonical = resolver.resolve("file_read")  # -> "read_file"

The resolver is integrated into ToolRegistry via a `resolver` parameter.
"""

from __future__ import annotations

import difflib
import json
import logging
import time
from typing import Any, Callable, Optional
from agent_framework.tools import ToolRegistry

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Resolution policies
# ---------------------------------------------------------------------------


class AliasPolicy:
    """Resolve via explicit alias mapping."""

    def __init__(self, aliases: dict[str, str]):
        self.aliases = aliases

    def resolve(self, name: str) -> Optional[str]:
        return self.aliases.get(name)

    @property
    def name(self) -> str:
        return "alias"


class FuzzyPolicy:
    """Resolve via fuzzy string matching (Levenshtein/difflib)."""

    def __init__(self, threshold: float = 0.8):
        self.threshold = threshold

    def resolve(self, name: str, registry: ToolRegistry) -> Optional[str]:
        candidates = difflib.get_close_matches(
            name, list(registry.tools.keys()), n=1, cutoff=self.threshold
        )
        if candidates:
            return candidates[0]
        return None

    @property
    def name(self) -> str:
        return "fuzzy"


class LLMPolicy:
    """Resolve via an LLM call (for hard cases).

    In practice this would use a lightweight model or cached embeddings.
    For now, this is a stub that logs and returns None.
    """

    def __init__(self, resolve_fn):
        self.resolve_fn = resolve_fn

    def resolve(self, name: str, registry: ToolRegistry) -> Optional[str]:
        if self.resolve_fn:
            result = self.resolve_fn(name, list(registry.tools.keys()))
            if result and result in registry.tools:
                return result
        return None

    @property
    def name(self) -> str:
        return "llm"


# ---------------------------------------------------------------------------
# ToolResolver
# ---------------------------------------------------------------------------


class ToolResolver:
    """Pluggable tool name resolution pipeline.

    Policies are tried in order. First match wins.
    """

    # Common default aliases — models love these
    DEFAULT_ALIASES: dict[str, str] = {
        # Read file
        "file_read": "read_file",
        "read": "read_file",
        "cat": "read_file",
        "read_text": "read_file",
        # Write file
        "file_write": "write_file",
        "write": "write_file",
        "create_file": "write_file",
        # Patch
        "patch": "patch_file",
        "edit": "patch_file",
        "edit_file": "patch_file",
        "replace": "patch_file",
        # Search
        "search": "code_search",
        "search_code": "code_search",
        "grep": "code_search",
        "codegrep": "code_search",
        "grep_search": "code_search",
        # Find files
        "find": "find_files",
        "list_files": "find_files",
        "ls": "find_files",
        "list_directory": "find_files",
        "dir": "find_files",
        # Terminal
        "run": "execute_terminal",
        "shell": "execute_terminal",
        "bash": "execute_terminal",
        "run_command": "execute_terminal",
        "exec": "execute_terminal",
        "terminal": "execute_terminal",
        # Python
        "python": "execute_python",
        "run_python": "execute_python",
        "exec_python": "execute_python",
        # Git
        "git": "git_status",
        "status": "git_status",
        "diff": "git_diff",
        "log": "git_log",
        "gitlog": "git_log",
    }

    def __init__(
        self,
        aliases: Optional[dict[str, str]] = None,
        registry: Optional[ToolRegistry] = None,
        llm_resolve_fn=None,
    ):
        # Build policies chain
        self.policies = []

        # 1. Aliases
        alias_map = dict(self.DEFAULT_ALIASES)
        if aliases:
            alias_map.update(aliases)
        self.policies.append(AliasPolicy(alias_map))

        # 2. Fuzzy
        self.policies.append(FuzzyPolicy(threshold=0.8))

        # 3. LLM (optional)
        if llm_resolve_fn:
            self.policies.append(LLMPolicy(llm_resolve_fn))

        # Cache for speed
        self._registry = registry
        self._cache: dict[str, Optional[str]] = {}
        self._miss_cache: dict[str, float] = {}  # negative cache with timestamp

    def resolve(self, name: str) -> Optional[str]:
        """Resolve a tool name through the policy chain.

        Returns the canonical registered name or None.
        """
        # Check cache first
        if name in self._cache:
            cached = self._cache[name]
            if cached and (self._registry is None or cached in self._registry.tools):
                return cached
            return None  # negative cache hit

        # P3 fix: Check negative cache (with TTL)
        if name in self._miss_cache:
            if time.monotonic() - self._miss_cache[name] < 300:  # 5 min TTL
                return None
            else:
                del self._miss_cache[name]  # expired

        # Try each policy
        for policy in self.policies:
            if isinstance(policy, AliasPolicy):
                result = policy.resolve(name)
            else:
                result = policy.resolve(name, self._registry)

            if result:
                # Verify it exists in registry
                if self._registry and result in self._registry.tools:
                    self._cache[name] = result
                    if result != name:
                        logger.info("Tool resolved: %s -> %s (via %s)", name, result, policy.name)
                    return result

        # P3 fix: Cache misses (negative cache)
        self._miss_cache[name] = time.monotonic()
        return None

    def set_registry(self, registry: ToolRegistry) -> None:
        """Set the registry for validation."""
        self._registry = registry
        self._cache.clear()

    def add_aliases(self, aliases: dict[str, str]) -> None:
        """Add more aliases at runtime."""
        if self.policies:
            self.policies[0].aliases.update(aliases)
            self._cache.clear()


# ---------------------------------------------------------------------------
# Convenience: build_default_resolver
# ---------------------------------------------------------------------------


def build_default_resolver(
    aliases: Optional[dict[str, str]] = None,
    registry: Optional[ToolRegistry] = None,
    llm_resolve_fn=None,
) -> ToolResolver:
    """Build a default resolver with all standard aliases and fuzzy matching.

    Args:
        aliases: Extra aliases to merge with defaults.
        registry: Optional registry for validation.
        llm_resolve_fn: Optional function(requested_name, available_names) -> canonical_name.

    Returns:
        Configured ToolResolver instance.
    """
    resolver = ToolResolver(aliases=aliases, registry=registry, llm_resolve_fn=llm_resolve_fn)
    return resolver

