"""
Optional LLM layer — provider-agnostic.

The user supplies base_url + api_key + model. We POST an OpenAI-compatible
chat/completions request (the de-facto standard that Anthropic, OpenAI, Together,
Groq, OpenRouter, local Ollama, etc. all accept at /v1/chat/completions or a
compatible path).

Used to give a semantic verdict on a flagged group: are the aligned spans truly
inconsistent translations of the same term, or acceptable contextual variants?
"""

from __future__ import annotations
import json
import urllib.request
import urllib.error
from typing import List, Dict, Optional


class LLMConfig:
    def __init__(self, base_url: str, api_key: str, model: str,
                 timeout: int = 60, extra_headers: Optional[Dict] = None):
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.model = model
        self.timeout = timeout
        self.extra_headers = extra_headers or {}


SYSTEM_PROMPT = (
    "You are a bilingual English-Arabic terminology QA assistant. "
    "Given an English term and the distinct Arabic translations found for it, "
    "judge whether they are genuinely inconsistent (the same term rendered "
    "differently and should be unified) or acceptable variants (grammatical "
    "inflection, valid synonyms in context). Respond ONLY with JSON: "
    '{"verdict":"inconsistent|acceptable","preferred":"<arabic>","reason":"<short>"}'
)


def _endpoint(base_url: str) -> str:
    # accept either a full endpoint or a base that needs the standard path
    if base_url.endswith("/chat/completions"):
        return base_url
    if base_url.endswith("/v1"):
        return base_url + "/chat/completions"
    return base_url + "/v1/chat/completions"


def judge_group(cfg: LLMConfig, ngram: str, spans: List[str]) -> Dict:
    """Ask the model to judge one flagged group. Returns parsed JSON or an error dict."""
    user = (
        f"English term: \"{ngram}\"\n"
        f"Distinct Arabic translations found:\n"
        + "\n".join(f"- {s}" for s in spans)
    )
    payload = {
        "model": cfg.model,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user},
        ],
        "temperature": 0,
        "max_tokens": 300,
    }
    data = json.dumps(payload).encode("utf-8")
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {cfg.api_key}",
    }
    headers.update(cfg.extra_headers)

    req = urllib.request.Request(_endpoint(cfg.base_url), data=data, headers=headers)
    try:
        with urllib.request.urlopen(req, timeout=cfg.timeout) as resp:
            body = json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as e:
        return {"error": f"HTTP {e.code}", "detail": e.read().decode("utf-8", "ignore")[:400]}
    except Exception as e:  # noqa
        return {"error": str(e)}

    try:
        content = body["choices"][0]["message"]["content"].strip()
        # strip code fences if present
        if content.startswith("```"):
            content = content.strip("`")
            content = content.split("\n", 1)[1] if "\n" in content else content
            if content.lower().startswith("json"):
                content = content.replace("json", "", 1).strip()
        return json.loads(content)
    except Exception:
        return {"verdict": "unknown", "raw": body.get("choices", [{}])[0]
                .get("message", {}).get("content", "")[:400]}


def test_connection(cfg: LLMConfig) -> Dict:
    """Lightweight check that credentials + endpoint work."""
    res = judge_group(cfg, "test", ["اختبار", "تجربة"])
    if "error" in res:
        return {"ok": False, **res}
    return {"ok": True, "sample": res}
