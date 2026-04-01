from __future__ import annotations

import os

MAX_OCR_CHARS = int(os.environ.get("LLM_OCR_MAX_CHARS", "9000"))


def _clip_ocr(ocr_text: str, max_chars: int | None = None) -> str:
    limit = max_chars if max_chars is not None else MAX_OCR_CHARS
    if len(ocr_text) > limit:
        return ocr_text[:limit] + "\n[... OCR truncado para economizar RAM/tempo ...]"
    return ocr_text


def build_quiz_extract_prompt(ocr_text: str, max_chars: int | None = None) -> str:
    """Passo 1 (2-pass): só extrai perguntas/opções em JSON (sem suggested)."""
    clipped = _clip_ocr(ocr_text, max_chars)
    return f"""You are a precise assistant. The text below was extracted with OCR from a screen and contains noise.

Task:
- Extract ONLY multiple-choice quiz blocks: full question and each option.
- Reconstruct OCR artifacts (merge split words/lines, fix obvious OCR mistakes).
- Number options sequentially (1,2,3...) if the original numbering is missing/garbled.

Output:
- Return a single JSON object only. No markdown. No extra text.
- Exact shape:
{{"items":[{{"question":"...","options":["opt1","opt2","opt3"]}}]}}

OCR:
---
{clipped}
---
"""


def build_quiz_suggest_prompt(extracted_json: str) -> str:
    """Passo 2 (2-pass): recebe JSON estruturado e retorna suggested/confidence/note."""
    return f"""You are a precise assistant. You will receive a JSON with multiple-choice questions and options.

Rules:
- For each item, set "suggested" to exactly ONE label: "A","B","C","D","E","F","G","H" or "1","2","3","4","5","6","7","8".
- confidence: "baixa", "media", or "alta".
- Optional short "note" (one line).
- WebSocket / real-time API questions: choose the option about live or bidirectional communication (e.g. chat), not batch jobs or static reports.

Output:
- Return a single JSON object only. No markdown. No extra text.
- Exact shape:
{{"items":[{{"suggested":"C","confidence":"media","note":""}}]}}
- The items list MUST have the same length and order as the input.

INPUT_JSON:
---
{extracted_json}
---
"""


def build_quiz_refine_prompt(ocr_text: str, max_chars: int | None = None) -> str:
    """Modo 1-pass: extrai perguntas/opções e já sugere alternativa em um JSON único."""
    clipped = _clip_ocr(ocr_text, max_chars)
    return f"""You are a precise assistant. The text below was extracted with OCR from a screen; it contains noise (browser tabs, sidebars, URLs, UI labels).

IMPORTANT — OCR text reconstruction:
- The OCR text may have broken words, split lines, or reordered fragments due to imperfect recognition.
- Before extracting questions and options, mentally reconstruct the original text: merge split words, rejoin broken lines, and fix obvious OCR artifacts (e.g. "l" instead of "1", "O" instead of "0").
- If an option appears split across two lines, merge them into a single option.
- Number options sequentially (1, 2, 3...) even if the original numbering is missing or garbled.

Rules:
- Ignore tabs, menus, sidebars, calendar items, URLs, and navigation fluff.
- Extract ONLY multiple-choice quiz blocks: the full question and each option line (same order as on screen).
- You MUST set "suggested" to exactly ONE label: "A","B","C","D","E","F","G","H" or "1","2","3","4","5","6","7","8" — pick the option that best answers the question by standard technical knowledge. Never leave "suggested" empty.
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

