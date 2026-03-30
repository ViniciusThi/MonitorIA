"""Cliente HTTP mínimo para Ollama (/api/generate). Sem dependências extras."""

from __future__ import annotations

import json
import os
import urllib.error
import urllib.request

DEFAULT_HOST = os.environ.get("OLLAMA_HOST", "http://127.0.0.1:11434").rstrip("/")
DEFAULT_MODEL = os.environ.get("OLLAMA_MODEL", "gemma3:4b")
DEFAULT_TIMEOUT = int(os.environ.get("OLLAMA_TIMEOUT", "120"))
MAX_OCR_CHARS = int(os.environ.get("OLLAMA_OCR_MAX_CHARS", "4500"))


def _resolved_num_predict(explicit: int | None, fallback: int = 768) -> int:
    if explicit is not None:
        return max(32, min(int(explicit), 16384))
    raw = os.environ.get("OLLAMA_NUM_PREDICT", "").strip()
    if raw == "":
        return max(32, min(fallback, 16384))
    try:
        return max(32, min(int(raw), 16384))
    except ValueError:
        return max(32, min(fallback, 16384))


def _ollama_options_extra() -> dict:
    raw = os.environ.get("OLLAMA_NUM_CTX", "2048").strip()
    if raw == "" or raw == "0":
        return {}
    try:
        n = int(raw)
    except ValueError:
        n = 2048
    return {"num_ctx": n} if n > 0 else {}


def _connection_error_from_ollama_http(url: str, code: int, body: str) -> ConnectionError:
    b = body or ""
    if "requires more system memory" in b or "more system memory" in b:
        return ConnectionError(
            "Ollama: RAM insuficiente para este modelo. Opções: feche outros programas; "
            "instale um modelo menor (`ollama pull gemma3:4b` ou `llama3.2:3b` ou `phi3:mini`) "
            "e defina OLLAMA_MODEL; ou reduza OLLAMA_NUM_CTX (ex.: 1024). "
            f"Resposta: {b[:350]}"
        )
    return ConnectionError(f"Ollama HTTP {code} em {url}. Corpo (trecho): {b[:450] or '(vazio)'}")


def _post_json(url: str, payload: dict, timeout: int) -> dict:
    body = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        url,
        data=body,
        method="POST",
        headers={"Content-Type": "application/json"},
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as e:
        body = ""
        try:
            body = e.read().decode("utf-8", errors="replace")
        except Exception:
            pass
        raise _connection_error_from_ollama_http(url, e.code, body) from e


def ollama_generate(
    prompt: str,
    *,
    base_url: str | None = None,
    model: str | None = None,
    num_predict: int | None = None,
    temperature: float = 0.2,
    timeout: int | None = None,
) -> str:
    """Chama POST /api/generate e devolve o campo 'response' (texto completo do modelo)."""
    base = (base_url or DEFAULT_HOST).rstrip("/")
    mod = model or DEFAULT_MODEL
    to = timeout if timeout is not None else DEFAULT_TIMEOUT
    url = f"{base}/api/generate"
    np = _resolved_num_predict(num_predict, 768)
    opt = {"temperature": temperature, "num_predict": np, **_ollama_options_extra()}
    payload: dict = {
        "model": mod,
        "prompt": prompt,
        "stream": False,
        "options": opt,
    }
    if os.environ.get("OLLAMA_FORMAT_JSON", "1").strip().lower() not in ("0", "false", "no"):
        payload["format"] = "json"
    try:
        data = _post_json(url, payload, to)
    except urllib.error.URLError as e:
        raise ConnectionError(
            f"Ollama não respondeu em {base}. Confirme se o serviço está rodando. Detalhe: {e}"
        ) from e
    except TimeoutError as e:
        raise TimeoutError(f"Ollama excedeu {to}s de espera.") from e
    if "response" not in data:
        raise ValueError(f"Resposta sem campo 'response': {data!r}")
    return str(data["response"])


def ollama_chat(
    prompt: str,
    *,
    base_url: str | None = None,
    model: str | None = None,
    num_predict: int | None = None,
    temperature: float = 0.2,
    timeout: int | None = None,
) -> str:
    """POST /api/chat — às vezes evita HTTP 500 que /api/generate dispara em algumas versões do Ollama."""
    base = (base_url or DEFAULT_HOST).rstrip("/")
    mod = model or DEFAULT_MODEL
    to = timeout if timeout is not None else DEFAULT_TIMEOUT
    url = f"{base}/api/chat"
    np = _resolved_num_predict(num_predict, 768)
    opt = {"temperature": temperature, "num_predict": np, **_ollama_options_extra()}
    payload = {
        "model": mod,
        "messages": [{"role": "user", "content": prompt}],
        "stream": False,
        "options": opt,
    }
    if os.environ.get("OLLAMA_FORMAT_JSON", "1").strip().lower() not in ("0", "false", "no"):
        payload["format"] = "json"
    try:
        data = _post_json(url, payload, to)
    except urllib.error.URLError as e:
        raise ConnectionError(
            f"Ollama não respondeu em {base}. Detalhe: {e}"
        ) from e
    except TimeoutError as e:
        raise TimeoutError(f"Ollama excedeu {to}s de espera.") from e
    msg = data.get("message")
    if isinstance(msg, dict) and "content" in msg:
        return str(msg["content"])
    raise ValueError(f"Resposta /api/chat sem message.content: {data!r}")


def ollama_complete(prompt: str, **kwargs) -> str:
    """Tenta /api/generate; em HTTP 500/502 tenta /api/chat com o mesmo prompt."""
    try:
        return ollama_generate(prompt, **kwargs)
    except ConnectionError as e:
        err = str(e)
        if "RAM insuficiente" in err:
            raise
        if "HTTP 500" in err or "HTTP 502" in err:
            return ollama_chat(prompt, **kwargs)
        raise


def build_quiz_refine_prompt(ocr_text: str, max_chars: int | None = None) -> str:
    """Monta o prompt único: extrair perguntas/opções limpas + sugestão em JSON."""
    limit = max_chars if max_chars is not None else MAX_OCR_CHARS
    if len(ocr_text) > limit:
        clipped = ocr_text[:limit] + "\n[... OCR truncado para economizar RAM/tempo ...]"
    else:
        clipped = ocr_text
    return f"""You are a precise assistant. The text below was extracted with OCR from a screen; it contains noise (browser tabs, sidebars, URLs, UI labels).

Rules:
- Ignore tabs, menus, sidebars, calendar items, URLs, and navigation fluff.
- Extract ONLY multiple-choice quiz blocks: the full question and each option line (same order as on screen).
- You MUST set "suggested" to exactly ONE label: "A","B","C","D","E" or "1","2","3","4","5" — pick the option that best answers the question by standard technical knowledge. Never leave "suggested" empty.
- If you are less sure, still pick the most likely option and set confidence to "baixa".
- confidence: "baixa", "media", or "alta".
- Optional short "note" with one-line reasoning.
- WebSocket / real-time API questions: choose the option about live or bidirectional communication (e.g. chat), not batch jobs or static reports.

Output a single JSON object only. No markdown. No text before or after the JSON.

Exact shape:
{{"items":[{{"question":"...","options":["opt1","opt2"],"suggested":"C","confidence":"media","note":""}}]}}

OCR:
---
{clipped}
---
"""


def parse_items_from_llm_response(raw: str) -> tuple[dict | None, str | None]:
    """Retorna (objeto com chave 'items', None) ou (None, mensagem de erro)."""
    text = raw.strip()
    if "```" in text:
        start = text.find("{")
        end = text.rfind("}")
        if start != -1 and end != -1 and end > start:
            text = text[start : end + 1]
    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        start = text.find("{")
        end = text.rfind("}")
        if start == -1 or end == -1 or end <= start:
            return None, "Resposta do modelo não contém JSON válido."
        try:
            data = json.loads(text[start : end + 1])
        except json.JSONDecodeError as e:
            return None, f"JSON inválido na resposta do modelo: {e}"

    if not isinstance(data, dict):
        return None, "JSON raiz deve ser um objeto."
    items = data.get("items")
    if not isinstance(items, list):
        return None, "JSON deve conter 'items' como lista."
    return data, None
