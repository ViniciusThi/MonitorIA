"""Servidor HTTP para exibir resultados de captura OCR.

Padrão: 0.0.0.0 (acessível na rede local, ex.: celular).
Para restringir somente ao PC, defina WEB_LOCAL_ONLY=1.
"""

from __future__ import annotations

import json
import os
import re
import threading
from datetime import datetime
from http.server import HTTPServer, BaseHTTPRequestHandler
from pathlib import Path

_WEB_DIR = Path(__file__).resolve().parent / "web"


class CaptureStore:
    """Thread-safe store: guarda latest + histórico de capturas."""

    def __init__(self, max_history: int = 25):
        self._lock = threading.Lock()
        self._history: list[dict] = []
        self._max = max_history
        self._version = 0

    def publish(self, result: dict) -> None:
        with self._lock:
            self._version += 1
            entry = {
                "id": self._version,
                "timestamp": datetime.now().isoformat(timespec="seconds"),
                "data": _web_safe_result(result),
            }
            self._history.append(entry)
            if len(self._history) > self._max:
                self._history = self._history[-self._max:]

    def latest(self) -> dict | None:
        with self._lock:
            return dict(self._history[-1]) if self._history else None

    def history(self) -> list[dict]:
        with self._lock:
            return list(self._history)

    @property
    def version(self) -> int:
        with self._lock:
            return self._version


def _web_safe_result(result: dict) -> dict:
    """Extrai apenas os campos úteis para a UI web."""
    items: list[dict] = []
    qs = result.get("questions") or []
    opts_by_q = result.get("options_by_question") or []
    summaries = result.get("summaries") or []
    timings_ms = result.get("timings_ms") or {}
    backend = "airllm"
    incomplete = 0
    parsed = result.get("ollama_parsed") or None

    for i, q in enumerate(qs):
        opts = opts_by_q[i] if i < len(opts_by_q) else []
        summary = summaries[i] if i < len(summaries) else None
        item: dict = {"question": q, "options": opts}
        if summary and isinstance(summary, dict):
            item["question_short"] = summary.get("question_short", q)
            item["options_short"] = summary.get("options_short", opts)
            item["complete"] = summary.get("complete", len(opts) >= 2)
        else:
            item["question_short"] = (q[:200] + "...") if len(q) > 200 else q
            item["options_short"] = opts
            item["complete"] = len(opts) >= 2
        if not item.get("complete", True):
            incomplete += 1
        items.append(item)

    return {
        "total_questions": result.get("total_questions", len(qs)),
        "items": items,
        "filtered_text_preview": (result.get("filtered_text") or "")[:3000],
        "backend": backend,
        "incomplete_questions": incomplete,
        "timings_ms": timings_ms,
        "parsed_json": parsed,
    }


class _WebHandler(BaseHTTPRequestHandler):
    store: CaptureStore | None = None

    def log_message(self, format, *args):  # noqa: A002
        pass

    def do_GET(self):
        if self.path == "/api/latest":
            self._json_response(self.store.latest())
        elif self.path == "/api/history":
            self._json_response(self.store.history())
        elif self.path.startswith("/api/version"):
            self._json_response({"version": self.store.version})
        elif self.path in ("/", "/index.html"):
            self._serve_file("index.html", "text/html; charset=utf-8")
        elif self.path == "/favicon.ico":
            self.send_error(204)
        else:
            self.send_error(404)

    def _json_response(self, data):
        body = json.dumps(data, ensure_ascii=False, default=str).encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-cache")
        self.end_headers()
        self.wfile.write(body)

    def _serve_file(self, filename: str, content_type: str):
        fpath = _WEB_DIR / filename
        if not fpath.is_file():
            self.send_error(404, f"{filename} not found")
            return
        body = fpath.read_bytes()
        self.send_response(200)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


def _get_local_ip() -> str:
    """Tenta descobrir o IP da máquina na rede local (para exibir ao usuário)."""
    import socket
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))
        ip = s.getsockname()[0]
        s.close()
        return ip
    except Exception:
        return "???"


def start_web_server(
    store: CaptureStore,
    port: int | None = None,
) -> tuple[HTTPServer, int, str]:
    """Inicia servidor HTTP em daemon thread. Retorna (server, porta, endereço de acesso).

    Padrão           → bind em 0.0.0.0 (acessível na rede local / celular).
    WEB_LOCAL_ONLY=1 → bind em 127.0.0.1 (somente este PC).
    """
    p = port or int(os.environ.get("WEB_PORT", "8765"))
    local_only = os.environ.get("WEB_LOCAL_ONLY", "").strip().lower() in ("1", "true", "yes", "on")
    bind_addr = "127.0.0.1" if local_only else "0.0.0.0"
    _WebHandler.store = store
    server = HTTPServer((bind_addr, p), _WebHandler)
    t = threading.Thread(target=server.serve_forever, daemon=True, name="web-server")
    t.start()

    local_ip = _get_local_ip()
    if local_only:
        access_url = f"http://127.0.0.1:{p}"
        print(f"  Web (somente local): {access_url}")
    else:
        access_url = f"http://{local_ip}:{p}"
        print(f"  Web (rede local):    {access_url}")
        print(f"  No celular, abra:    http://{local_ip}:{p}")
        print(f"  Se nao abrir, verifique o firewall do Windows (porta {p} TCP).")

    return server, p, access_url
