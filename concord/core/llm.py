"""
Optional LLM layer — provider-agnostic.

Supports two request shapes:
  - OpenAI-compatible /v1/chat/completions (OpenAI, Together, Groq, OpenRouter,
    local Ollama, and most proxies).
  - Anthropic native /v1/messages (Claude directly, no proxy needed).

The shape is auto-detected from the base URL (or forced via provider=). Used to
give a semantic verdict on a flagged group: are the aligned spans genuinely
inconsistent, or acceptable contextual variants? judge_all runs a whole batch
of groups concurrently.
"""

from __future__ import annotations
import json
import urllib.request
import urllib.error
from concurrent.futures import ThreadPoolExecutor
from typing import List, Dict, Optional

ANTHROPIC_VERSION = "2023-06-01"


class LLMConfig:
    def __init__(self, base_url: str, api_key: str, model: str,
                 timeout: int = 60, provider: str = "auto",
                 extra_headers: Optional[Dict] = None):
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.model = model
        self.timeout = timeout
        self.provider = provider
        self.extra_headers = extra_headers or {}

    def is_anthropic(self) -> bool:
        if self.provider == "anthropic":
            return True
        if self.provider == "openai":
            return False
        return "anthropic.com" in self.base_url


SYSTEM_PROMPT = (
    "You are a bilingual English-Arabic terminology QA assistant. "
    "Given an English term and the distinct Arabic translations found for it, "
    "judge whether they are genuinely inconsistent (the same term rendered "
    "differently and should be unified) or acceptable variants (grammatical "
    "inflection, valid synonyms in context). Respond ONLY with JSON: "
    '{"verdict":"inconsistent|acceptable","preferred":"<arabic>","reason":"<short>"}'
)


def _openai_endpoint(base_url: str) -> str:
    if base_url.endswith("/chat/completions"):
        return base_url
    if base_url.endswith("/v1"):
        return base_url + "/chat/completions"
    return base_url + "/v1/chat/completions"


def _anthropic_endpoint(base_url: str) -> str:
    if base_url.endswith("/messages"):
        return base_url
    if base_url.endswith("/v1"):
        return base_url + "/messages"
    return base_url + "/v1/messages"


def _user_prompt(ngram: str, spans: List[str]) -> str:
    return (
        f'English term: "{ngram}"\n'
        "Distinct Arabic translations found:\n"
        + "\n".join(f"- {s}" for s in spans)
    )


def _post(url: str, payload: dict, headers: dict, timeout: int) -> Dict:
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(url, data=data, headers=headers)
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as e:
        detail = e.read().decode("utf-8", "ignore")[:400]
        return {"__error__": f"HTTP {e.code}", "detail": detail}
    except Exception as e:  # noqa
        return {"__error__": str(e)}


def _parse_verdict(content: str, raw) -> Dict:
    content = (content or "").strip()
    if content.startswith("```"):
        content = content.strip("`")
        content = content.split("\n", 1)[1] if "\n" in content else content
        if content.lower().startswith("json"):
            content = content.replace("json", "", 1).strip()
    try:
        return json.loads(content)
    except Exception:
        return {"verdict": "unknown", "raw": (content or str(raw))[:400]}


def judge_group(cfg: LLMConfig, ngram: str, spans: List[str]) -> Dict:
    """Ask the model to judge one flagged group. Returns parsed JSON or error."""
    user = _user_prompt(ngram, spans)
    if cfg.is_anthropic():
        headers = {
            "Content-Type": "application/json",
            "x-api-key": cfg.api_key,
            "anthropic-version": ANTHROPIC_VERSION,
        }
        headers.update(cfg.extra_headers)
        payload = {
            "model": cfg.model,
            "max_tokens": 300,
            "system": SYSTEM_PROMPT,
            "messages": [{"role": "user", "content": user}],
        }
        body = _post(_anthropic_endpoint(cfg.base_url), payload, headers, cfg.timeout)
        if "__error__" in body:
            return {"error": body["__error__"], "detail": body.get("detail", "")}
        parts = body.get("content", [])
        text = parts[0].get("text", "") if parts else ""
        return _parse_verdict(text, body)

    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {cfg.api_key}",
    }
    headers.update(cfg.extra_headers)
    payload = {
        "model": cfg.model,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user},
        ],
        "temperature": 0,
        "max_tokens": 300,
    }
    body = _post(_openai_endpoint(cfg.base_url), payload, headers, cfg.timeout)
    if "__error__" in body:
        return {"error": body["__error__"], "detail": body.get("detail", "")}
    try:
        text = body["choices"][0]["message"]["content"]
    except Exception:
        text = ""
    return _parse_verdict(text, body)


def judge_all(cfg: LLMConfig, items: List[Dict], max_workers: int = 4) -> List[Dict]:
    """Judge many groups concurrently. items: [{"ngram":..,"spans":[..]}].
    Returns a verdict dict per item, in the same order."""
    def one(it):
        return judge_group(cfg, it["ngram"], it["spans"])

    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        return list(ex.map(one, items))


def test_connection(cfg: LLMConfig) -> Dict:
    """Lightweight check that credentials + endpoint work."""
    res = judge_group(cfg, "test", ["اختبار", "تجربة"])
    if "error" in res:
        return {"ok": False, **res}
    return {"ok": True, "sample": res}
