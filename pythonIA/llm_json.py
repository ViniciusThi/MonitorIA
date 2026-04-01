from __future__ import annotations

import json


def parse_items_from_llm_response(raw: str) -> tuple[dict | None, str | None]:
    """Retorna (objeto com chave 'items', None) ou (None, mensagem de erro)."""
    text = (raw or "").strip()
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

