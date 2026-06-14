from __future__ import annotations

import json
import urllib.error
import urllib.request
from typing import Any

from .prompting import PHYSICS_SYSTEM_PROMPT


def normalize_base_url(base_url: str) -> str:
    base = base_url.rstrip("/")
    if not base.endswith("/v1"):
        base = f"{base}/v1"
    return base


def http_json(method: str, url: str, payload: dict[str, Any] | None = None, timeout: float = 60.0) -> dict[str, Any]:
    data = None
    headers = {"Accept": "application/json"}
    if payload is not None:
        data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        headers["Content-Type"] = "application/json"
    request = urllib.request.Request(url, data=data, headers=headers, method=method)
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            body = response.read().decode("utf-8")
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"HTTP {exc.code} from {url}: {detail}") from exc
    except urllib.error.URLError as exc:
        raise RuntimeError(f"Cannot reach {url}: {exc}") from exc
    return json.loads(body)


def check_model(base_url: str, model: str, timeout: float) -> None:
    response = http_json("GET", f"{base_url}/models", timeout=timeout)
    model_ids = {item.get("id") for item in response.get("data", []) if isinstance(item, dict)}
    if model not in model_ids:
        available = ", ".join(sorted(str(item) for item in model_ids if item)) or "<none>"
        raise RuntimeError(
            f"Model {model!r} was not found at /v1/models. Available models: {available}. "
            "Use --skip-model-check only if your server hides model metadata."
        )


def call_qwen(
    *,
    base_url: str,
    model: str,
    prompt: str,
    timeout: float,
    temperature: float,
    max_tokens: int,
    disable_thinking: bool,
    use_response_format: bool,
) -> str:
    user_prompt = prompt
    if disable_thinking and not user_prompt.lstrip().startswith("/no_think"):
        user_prompt = "/no_think\n" + user_prompt
    payload: dict[str, Any] = {
        "model": model,
        "messages": [
            {"role": "system", "content": PHYSICS_SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ],
        "temperature": temperature,
        "max_tokens": max_tokens,
    }
    if use_response_format:
        payload["response_format"] = {"type": "json_object"}
    response = http_json("POST", f"{base_url}/chat/completions", payload=payload, timeout=timeout)
    try:
        return str(response["choices"][0]["message"]["content"])
    except (KeyError, IndexError, TypeError) as exc:
        raise RuntimeError(f"Unexpected chat completion response: {response}") from exc

