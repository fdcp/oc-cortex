"""
Shared LLM client helper.

Resolves an OpenAI client for any model declared in
config/opencode_models.yaml. The endpoint name drives required headers:

- "go"   → x-opencode-session (paid, OpenCode Go balance)
- "zen"  → User-Agent: opencode/<ver> (free tier, Console upstream gate)
- other  → no special headers (e.g., minimax, other OpenAI-compatible)

Only a model name is required. To extend a model or add a provider,
edit config/opencode_models.yaml.
"""
import json
import os
import time
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Optional

import yaml
from loguru import logger
from openai import OpenAI


DEFAULT_OPENCODE_VERSION = "1.18.18"
DEFAULT_MODEL = "MiniMax-M3"
DEFAULT_YAML_PATH = "config/opencode_models.yaml"
DEFAULT_AUTH_FILE = "~/.local/share/opencode/auth.json"
FALLBACK_API_KEY_ENV = "OPENCODE_ZEN_API_KEY"


class LLMProvider(str, Enum):
    GO = "go"
    ZEN = "zen"
    CUSTOM = "custom"


@dataclass
class ResolvedEndpoint:
    name: str
    provider: LLMProvider
    base_url: str
    api_key: str


def _provider_for_endpoint(endpoint_name: str) -> LLMProvider:
    try:
        return LLMProvider(endpoint_name)
    except ValueError:
        return LLMProvider.CUSTOM


def _resolve_yaml_path(yaml_path: str) -> Optional[Path]:
    p = Path(yaml_path)
    if p.is_absolute():
        return p if p.exists() else None
    candidates = [Path.cwd() / p, Path(__file__).resolve().parents[1] / p]
    return next((c for c in candidates if c.exists()), None)


def resolve_endpoint(
    model_name: str,
    yaml_path: str = DEFAULT_YAML_PATH,
) -> Optional[ResolvedEndpoint]:
    p = _resolve_yaml_path(yaml_path)
    if p is None:
        logger.warning(f"opencode_models.yaml 找不到: {yaml_path}")
        return None

    try:
        cfg = yaml.safe_load(p.read_text(encoding="utf-8")) or {}
    except (yaml.YAMLError, OSError) as e:
        logger.warning(f"加载 {p} 失败: {e}")
        return None

    model_entry = next(
        (m for m in cfg.get("models", []) if m.get("name") == model_name),
        None,
    )
    if model_entry is None:
        logger.warning(f"模型 {model_name!r} 不在 {p}")
        return None

    endpoint_name = model_entry.get("endpoint", "")
    endpoint = cfg.get("endpoints", {}).get(endpoint_name, {})
    base_url = endpoint.get("base_url", "")

    auth_file_rel = cfg.get("auth_file", DEFAULT_AUTH_FILE)
    api_key = ""
    auth_path = Path(auth_file_rel).expanduser()
    if auth_path.exists():
        try:
            auth_data = json.loads(auth_path.read_text(encoding="utf-8"))
            auth_provider = endpoint.get("auth_provider", "opencode-go")
            api_key = (auth_data.get(auth_provider, {}).get("key", "") or "")
        except (json.JSONDecodeError, KeyError) as e:
            logger.warning(f"读取 {auth_path} 失败: {e}")
    else:
        logger.warning(f"auth.json 不存在: {auth_file_rel}")

    if not api_key:
        api_key = os.environ.get(FALLBACK_API_KEY_ENV, "")

    return ResolvedEndpoint(
        name=endpoint_name,
        provider=_provider_for_endpoint(endpoint_name),
        base_url=base_url,
        api_key=api_key,
    )


def get_client(
    model_name: str = DEFAULT_MODEL,
    yaml_path: str = DEFAULT_YAML_PATH,
    api_key_override: Optional[str] = None,
    base_url_override: Optional[str] = None,
) -> OpenAI:
    ep = resolve_endpoint(model_name, yaml_path)
    if ep is None:
        raise ValueError(f"无法解析模型 {model_name!r} 的 endpoint")

    base_url = base_url_override or ep.base_url
    api_key = api_key_override or ep.api_key
    if not api_key:
        raise ValueError(f"模型 {model_name!r} 没有可用的 API key")

    headers: dict = {}
    if ep.provider == LLMProvider.GO:
        headers["x-opencode-session"] = (
            os.environ.get("OPENCODE_SESSION_ID")
            or f"cli-{os.getpid()}-{int(time.time())}"
        )
    elif ep.provider == LLMProvider.ZEN:
        headers["User-Agent"] = (
            os.environ.get("OPENCODE_USER_AGENT")
            or f"opencode/{os.environ.get('OPENCODE_VERSION', DEFAULT_OPENCODE_VERSION)}"
        )
        headers["x-opencode-session"] = (
            os.environ.get("OPENCODE_SESSION_ID")
            or f"cli-{os.getpid()}-{int(time.time())}"
        )

    return OpenAI(
        api_key=api_key,
        base_url=base_url,
        timeout=120,
        default_headers=headers or None,
    )