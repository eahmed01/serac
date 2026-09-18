from __future__ import annotations

from pathlib import Path

import pytest
import yaml

import agent_framework.model_targets as model_targets


def write_config(tmp_path: Path, targets: dict) -> Path:
    path = tmp_path / "targets.yaml"
    path.write_text(yaml.safe_dump({"version": 1, "targets": targets}, sort_keys=False))
    return path


def test_default_config_resolves_local_target(tmp_path):
    path = write_config(tmp_path, {
        "local_qwen": {
            "provider": "vllm",
            "endpoint": "http://127.0.0.1:7999/v1",
            "model": "Qwen/Qwen3.8-27B-FP8",
            "max_tokens": 24576,
            "vllm": {
                "reasoning_effort": "xhigh",
                "chat_template_kwargs": {
                    "enable_thinking": True,
                    "preserve_thinking": False,
                },
            },
        },
    })
    target = model_targets.resolve_model_target(path, "local_qwen", environ={})
    assert target.provider == "vllm"
    assert target.model == "Qwen/Qwen3.8-27B-FP8"
    assert target.endpoint == "http://127.0.0.1:7999/v1"
    assert target.manifest()["logical_target"] == "local_qwen"
    assert target.max_tokens == 24576
    assert target.vllm_options == {
        "reasoning_effort": "xhigh",
        "chat_template_kwargs": {
            "enable_thinking": True,
            "preserve_thinking": False,
        },
    }


def test_openai_target_requires_named_credential_without_exposing_value(tmp_path):
    path = write_config(tmp_path, {
        "openai_test": {
            "provider": "openai",
            "model": "gpt-5.6-luna",
            "credential_env": "TEST_OPENAI_KEY",
        },
    })
    with pytest.raises(model_targets.ModelTargetError, match="credential_env 'TEST_OPENAI_KEY'") as exc:
        model_targets.resolve_model_target(path, "openai_test", environ={})
    assert "sentinel-secret" not in str(exc.value)

    target = model_targets.resolve_model_target(path, "openai_test", environ={"TEST_OPENAI_KEY": "sentinel-secret"})
    assert "sentinel-secret" not in repr(target)
    assert "sentinel-secret" not in str(target.manifest())


def test_unknown_and_incompatible_target_configuration_fails(tmp_path):
    unknown = write_config(tmp_path, {
        "bad": {"provider": "vllm", "model": "x", "endpoint": "http://localhost/v1", "mystery": True},
    })
    with pytest.raises(model_targets.ModelTargetError, match="unknown keys"):
        model_targets.load_model_targets(unknown)

    incompatible = write_config(tmp_path, {
        "bad": {"provider": "openai", "model": "x", "credential_env": "KEY", "vllm": {}},
    })
    with pytest.raises(model_targets.ModelTargetError, match="vllm settings"):
        model_targets.load_model_targets(incompatible)


def test_factory_maps_vllm_settings(monkeypatch, tmp_path):
    path = write_config(tmp_path, {
        "local": {
            "provider": "vllm",
            "endpoint": "http://localhost:7999/v1/",
            "model": "local-model",
            "max_tokens": 1234,
            "credential_env": "TEST_VLLM_KEY",
            "vllm": {
                "reasoning_effort": None,
                "chat_template_kwargs": {"enable_thinking": False},
            },
        },
    })
    target = model_targets.resolve_model_target(path, "local", environ={"TEST_VLLM_KEY": "sentinel-secret"})
    captured = {}

    class FakeVLLM:
        def __init__(self, **kwargs): captured.update(kwargs)

    monkeypatch.setattr(model_targets, "VLLMProvider", FakeVLLM)
    model_targets.make_provider(target, environ={"TEST_VLLM_KEY": "sentinel-secret"})
    assert captured == {
        "base_url": "http://localhost:7999/v1",
        "model": "local-model",
        "max_tokens": 1234,
        "reasoning_effort": None,
        "chat_template_kwargs": {"enable_thinking": False},
        "api_key": "sentinel-secret",
    }


def test_factory_maps_openai_settings_and_keeps_key_out_of_target(monkeypatch, tmp_path):
    path = write_config(tmp_path, {
        "openai": {
            "provider": "openai",
            "model": "gpt-5.6-luna",
            "max_tokens": 2048,
            "credential_env": "TEST_OPENAI_KEY",
            "openai": {"base_url": "https://api.example.test/v1", "max_tokens_parameter": "max_completion_tokens"},
        },
    })
    target = model_targets.resolve_model_target(path, "openai", environ={"TEST_OPENAI_KEY": "sentinel-secret"})
    captured = {}

    class FakeOpenAI:
        def __init__(self, **kwargs): captured.update(kwargs)

    monkeypatch.setattr(model_targets, "OpenAIProvider", FakeOpenAI)
    model_targets.make_provider(target, environ={"TEST_OPENAI_KEY": "sentinel-secret"})
    assert captured == {
        "api_key": "sentinel-secret",
        "model": "gpt-5.6-luna",
        "max_tokens": 2048,
        "max_tokens_parameter": "max_completion_tokens",
        "base_url": "https://api.example.test/v1",
    }
    assert "sentinel-secret" not in str(target.manifest())


def test_factory_maps_openai_responses_settings(monkeypatch, tmp_path):
    path = write_config(tmp_path, {
        "openai": {
            "provider": "openai",
            "model": "gpt-5.6-luna",
            "credential_env": "TEST_OPENAI_KEY",
            "openai": {
                "api_mode": "responses",
                "reasoning_effort": "medium",
            },
        },
    })
    target = model_targets.resolve_model_target(path, "openai", environ={"TEST_OPENAI_KEY": "sentinel-secret"})
    captured = {}

    class FakeOpenAI:
        def __init__(self, **kwargs): captured.update(kwargs)

    monkeypatch.setattr(model_targets, "OpenAIProvider", FakeOpenAI)
    model_targets.make_provider(target, environ={"TEST_OPENAI_KEY": "sentinel-secret"})
    assert captured == {
        "api_key": "sentinel-secret",
        "model": "gpt-5.6-luna",
        "max_tokens": 4096,
        "max_tokens_parameter": "max_tokens",
        "api_mode": "responses",
        "reasoning_effort": "medium",
    }
