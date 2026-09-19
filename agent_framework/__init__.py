"""Agent Framework — standalone multi-agent orchestration library.

Three-tier architecture:
- Tier 1 (Orchestrator): User-facing agent
- Tier 2 (Worker Manager): Task decomposition + parallel dispatch
- Tier 3 (Workers): Do actual work, dispatched from a model pool

Phase 1: Core components (loop, providers, tools, memory, sandbox).
"""

from agent_framework.loop import AgentLoop
from agent_framework.providers import (
    AnthropicProvider,
    ChatResponse,
    OpenAIProvider,
    Provider,
    UsageStats,
    VLLMProvider,
)
from agent_framework.tools import ToolDef, ToolRegistry
from agent_framework.memory import MemoryStore

# sandbox and model_pool modules are not yet created — import conditionally
try:
    from agent_framework.docker_sandbox import Sandbox, SandboxError
except ImportError:
    Sandbox = SandboxError = None  # type: ignore[assignment,misc]

try:
    from agent_framework.worker_manager import WorkerManager, WorkerTask, WorkerResult
except ImportError:
    WorkerManager = WorkerTask = WorkerResult = None  # type: ignore[assignment,misc]

try:
    from agent_framework.model_pool import ModelPool, PoolRoutingResult, ProviderSlot
except ImportError:
    ModelPool = PoolRoutingResult = ProviderSlot = None  # type: ignore[assignment,misc]

from agent_framework.builtins import register_builtin_tools
from agent_framework.tool_loader import (
    load_default_tools,
    load_tools_from_config,
    ToolLoader,
)
from agent_framework.tool_resolver import (
    ToolResolver,
    AliasPolicy,
    FuzzyPolicy,
    LLMPolicy,
    build_default_resolver,
)
from agent_framework.retry import (
    ProviderError,
    RetryConfig,
    DEFAULT_RETRY,
    classify_exception,
    retry_call,
)
from agent_framework.tracing import (
    Tracer,
    Span,
    set_trace_id,
    get_trace_id,
    clear_trace,
)
from agent_framework.model_targets import (
    ModelTarget,
    ModelTargetError,
    SUPPORTED_VERSION,
    SUPPORTED_PROVIDERS,
    load_model_targets,
    resolve_model_target,
    make_provider,
)

__all__ = [
    "AgentLoop",
    "Provider",
    "VLLMProvider",
    "AnthropicProvider",
    "OpenAIProvider",
    "ToolRegistry",
    "ToolDef",
    "MemoryStore",
    "Sandbox",
    "SandboxError",
    "WorkerManager",
    "WorkerTask",
    "WorkerResult",
    "ModelPool",
    "PoolRoutingResult",
    "ProviderSlot",
    "register_builtin_tools",
    "load_default_tools",
    "load_tools_from_config",
    "ToolLoader",
    "ToolResolver",
    "AliasPolicy",
    "FuzzyPolicy",
    "LLMPolicy",
    "build_default_resolver",
    # Phase 4: retry + tracing
    "ProviderError",
    "RetryConfig",
    "DEFAULT_RETRY",
    "classify_exception",
    "retry_call",
    "Tracer",
    "Span",
    "set_trace_id",
    "get_trace_id",
    "clear_trace",
    # model targets
    "ModelTarget",
    "ModelTargetError",
    "SUPPORTED_VERSION",
    "SUPPORTED_PROVIDERS",
    "load_model_targets",
    "resolve_model_target",
    "make_provider",
]
