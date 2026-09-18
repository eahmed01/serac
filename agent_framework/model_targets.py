"""Logical model-target configuration and provider construction."""
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import yaml

from agent_framework.providers import OpenAIProvider, Provider, VLLMProvider

SUPPORTED_VERSION = 1
SUPPORTED_PROVIDERS = {"vllm", "openai"}
_TARGET_KEYS = {"provider", "endpoint", "model", "max_tokens", "credential_env", "vllm", "openai"}
_ROOT_KEYS = {"version", "targets"}
_VLLM_KEYS = {"reasoning_effort", "chat_template_kwargs"}
_OPENAI_KEYS = {"base_url", "max_tokens_parameter", "api_mode", "reasoning_effort"}
_OPENAI_TOKEN_PARAMETERS = {"max_tokens", "max_completion_tokens"}
_OPENAI_API_MODES = {"chat_completions", "responses"}
_REASONING_EFFORTS = {None, "low", "medium", "high", "xhigh"}


class ModelTargetError(ValueError):
    """Raised when a logical model target cannot be safely resolved."""


@dataclass(frozen=True)
class ModelTarget:
    """Validated, non-secret provider configuration."""

    name: str
    provider: str
    model: str
    max_tokens: int
    endpoint: str | None = None
    credential_env: str | None = None
    vllm_options: dict[str, Any] | None = None
    openai_options: dict[str, Any] | None = None

    def manifest(self) -> dict[str, Any]:
        """Return reproducible metadata without credential values."""
        result: dict[str, Any] = {
            "logical_target": self.name,
            "provider": self.provider,
            "model": self.model,
            "max_tokens": self.max_tokens,
        }
        if self.endpoint is not None:
            result["endpoint"] = self.endpoint
        if self.credential_env is not None:
            result["credential_env"] = self.credential_env
        if self.vllm_options:
            result["vllm"] = self.vllm_options
        if self.openai_options:
            result["openai"] = self.openai_options
        return result


def _mapping(value: Any, label: str) -> dict[str, Any]:
    if value is None:
        return {}
    if not isinstance(value, dict):
        raise ModelTargetError(f"{label} must be a mapping")
    return dict(value)


def _validate_common(name: str, raw: Mapping[str, Any]) -> None:
    unknown = set(raw) - _TARGET_KEYS
    if unknown:
        raise ModelTargetError(f"target '{name}' has unknown keys: {', '.join(sorted(unknown))}")
    provider = raw.get("provider")
    if provider not in SUPPORTED_PROVIDERS:
        raise ModelTargetError(f"target '{name}' provider must be one of: {', '.join(sorted(SUPPORTED_PROVIDERS))}")
    model = raw.get("model")
    if not isinstance(model, str) or not model.strip():
        raise ModelTargetError(f"target '{name}' model must be a non-empty string")
    max_tokens = raw.get("max_tokens", 4096)
    if isinstance(max_tokens, bool) or not isinstance(max_tokens, int) or max_tokens <= 0:
        raise ModelTargetError(f"target '{name}' max_tokens must be a positive integer")


def _target_from_raw(name: str, raw: Any) -> ModelTarget:
    if not isinstance(raw, dict):
        raise ModelTargetError(f"target '{name}' must be a mapping")
    _validate_common(name, raw)
    provider = raw["provider"]
    model = raw["model"].strip()
    max_tokens = raw.get("max_tokens", 4096)
    credential_env = raw.get("credential_env")
    if credential_env is not None and (not isinstance(credential_env, str) or not credential_env.strip()):
        raise ModelTargetError(f"target '{name}' credential_env must be a non-empty string")
    credential_env = credential_env.strip() if credential_env else None

    vllm_raw = _mapping(raw.get("vllm"), f"target '{name}' vllm")
    openai_raw = _mapping(raw.get("openai"), f"target '{name}' openai")
    if provider == "vllm":
        endpoint = raw.get("endpoint")
        if not isinstance(endpoint, str) or not endpoint.strip():
            raise ModelTargetError(f"target '{name}' vllm endpoint must be a non-empty string")
        if "openai" in raw:
            raise ModelTargetError(f"target '{name}' cannot define openai settings for provider vllm")
        unknown = set(vllm_raw) - _VLLM_KEYS
        if unknown:
            raise ModelTargetError(f"target '{name}' vllm has unknown keys: {', '.join(sorted(unknown))}")
        reasoning_effort = vllm_raw.get("reasoning_effort", "high")
        if reasoning_effort not in _REASONING_EFFORTS:
            raise ModelTargetError(f"target '{name}' vllm reasoning_effort is invalid")
        chat_template_kwargs = vllm_raw.get("chat_template_kwargs", {})
        if not isinstance(chat_template_kwargs, dict):
            raise ModelTargetError(f"target '{name}' vllm chat_template_kwargs must be a mapping")
        options = {
            "reasoning_effort": reasoning_effort,
            "chat_template_kwargs": dict(chat_template_kwargs),
        }
        return ModelTarget(name, provider, model, max_tokens, endpoint=endpoint.rstrip("/"), credential_env=credential_env, vllm_options=options)

    if raw.get("endpoint") is not None:
        raise ModelTargetError(f"target '{name}' openai endpoint must be configured under openai.base_url")
    if "vllm" in raw:
        raise ModelTargetError(f"target '{name}' cannot define vllm settings for provider openai")
    unknown = set(openai_raw) - _OPENAI_KEYS
    if unknown:
        raise ModelTargetError(f"target '{name}' openai has unknown keys: {', '.join(sorted(unknown))}")
    if not credential_env:
        raise ModelTargetError(f"target '{name}' requires credential_env for provider openai")
    base_url = openai_raw.get("base_url")
    if base_url is not None and (not isinstance(base_url, str) or not base_url.strip()):
        raise ModelTargetError(f"target '{name}' openai base_url must be a non-empty string when set")
    max_tokens_parameter = openai_raw.get("max_tokens_parameter", "max_tokens")
    if max_tokens_parameter not in _OPENAI_TOKEN_PARAMETERS:
        raise ModelTargetError(f"target '{name}' openai max_tokens_parameter is invalid")
    openai_options = {"max_tokens_parameter": max_tokens_parameter}
    api_mode = openai_raw.get("api_mode", "chat_completions")
    if api_mode not in _OPENAI_API_MODES:
        raise ModelTargetError(f"target '{name}' openai api_mode is invalid")
    if api_mode != "chat_completions":
        openai_options["api_mode"] = api_mode
    reasoning_effort = openai_raw.get("reasoning_effort")
    if reasoning_effort not in _REASONING_EFFORTS:
        raise ModelTargetError(f"target '{name}' openai reasoning_effort is invalid")
    if reasoning_effort is not None:
        openai_options["reasoning_effort"] = reasoning_effort
    if base_url:
        openai_options["base_url"] = base_url.rstrip("/")
    return ModelTarget(name, provider, model, max_tokens, credential_env=credential_env,
                       openai_options=openai_options)


def load_model_targets(path: str | Path) -> dict[str, ModelTarget]:
    """Load and validate all logical targets from a YAML file."""
    path = Path(path)
    try:
        raw = yaml.safe_load(path.read_text())
    except OSError as exc:
        raise ModelTargetError(f"model target config unavailable: {path}") from exc
    except yaml.YAMLError as exc:
        raise ModelTargetError(f"model target config is invalid YAML: {path}") from exc
    if not isinstance(raw, dict):
        raise ModelTargetError("model target config must be a mapping")
    unknown = set(raw) - _ROOT_KEYS
    if unknown:
        raise ModelTargetError(f"model target config has unknown keys: {', '.join(sorted(unknown))}")
    if raw.get("version") != SUPPORTED_VERSION:
        raise ModelTargetError(f"model target config version must be {SUPPORTED_VERSION}")
    targets = raw.get("targets")
    if not isinstance(targets, dict) or not targets:
        raise ModelTargetError("model target config targets must be a non-empty mapping")
    return {name: _target_from_raw(name, value) for name, value in targets.items()}


def resolve_model_target(path: str | Path, name: str, environ: Mapping[str, str] | None = None) -> ModelTarget:
    """Resolve one logical target and validate its credential availability."""
    targets = load_model_targets(path)
    if name not in targets:
        raise ModelTargetError(f"unknown model target '{name}'; available: {', '.join(sorted(targets))}")
    target = targets[name]
    environment = os.environ if environ is None else environ
    if target.credential_env and not environment.get(target.credential_env):
        raise ModelTargetError(f"credential_env '{target.credential_env}' is not set in the environment")
    return target


def make_provider(target: ModelTarget, environ: Mapping[str, str] | None = None) -> Provider:
    """Construct the configured provider without exposing credential values."""
    env = os.environ if environ is None else environ
    if target.provider == "vllm":
        options = target.vllm_options or {}
        provider_kwargs: dict[str, Any] = {
            "base_url": target.endpoint or "",
            "model": target.model,
            "max_tokens": target.max_tokens,
            "reasoning_effort": options.get("reasoning_effort"),
            "chat_template_kwargs": options.get("chat_template_kwargs"),
        }
        if target.credential_env:
            if not env.get(target.credential_env):
                raise ModelTargetError(f"credential_env '{target.credential_env}' is not set in the environment")
            provider_kwargs["api_key"] = env[target.credential_env]
        return VLLMProvider(**provider_kwargs)
    if target.provider == "openai":
        if not target.credential_env or not env.get(target.credential_env):
            raise ModelTargetError(f"credential_env '{target.credential_env}' is not set in the environment")
        return OpenAIProvider(
            api_key=env[target.credential_env],
            model=target.model,
            max_tokens=target.max_tokens,
            **(target.openai_options or {}),
        )
    raise ModelTargetError(f"unsupported provider '{target.provider}'")
