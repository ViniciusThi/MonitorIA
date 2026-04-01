import ctypes
import os
import re
import threading
import time
import tkinter as tk
from tkinter import ttk
import json
import webbrowser

import cv2
import keyboard
import mss
import numpy as np
import pytesseract
import win32gui

from local_web import CaptureStore, start_web_server
from prompts import build_quiz_extract_prompt, build_quiz_refine_prompt, build_quiz_suggest_prompt
from llm_json import parse_items_from_llm_response
from llm_providers import LLMUnavailableError
from airllm_client import AirLLMConfig, AirLLMProvider

# Tesseract: TESSERACT_CMD no ambiente ou caminho padrão no Windows
_tess_cmd = os.environ.get("TESSERACT_CMD")
if _tess_cmd:
    pytesseract.pytesseract.tesseract_cmd = _tess_cmd
elif os.name == "nt":
    _candidate_paths = [
        r"C:\Program Files\Tesseract-OCR\tesseract.exe",
        r"C:\Program Files (x86)\Tesseract-OCR\tesseract.exe",
        os.path.join(os.environ.get("LOCALAPPDATA", ""), r"Programs\Tesseract-OCR\tesseract.exe"),
    ]
    for _path in _candidate_paths:
        if _path and os.path.isfile(_path):
            pytesseract.pytesseract.tesseract_cmd = _path
            break

# PSM 6 costuma funcionar melhor para blocos com múltiplas linhas e opções (Canvas/LMS).
_TESSERACT_CONFIG = os.environ.get("TESSERACT_CONFIG", r"--oem 3 --psm 6")
_TESSERACT_LANG = os.environ.get("TESSERACT_LANG", "eng+por")

_OPTION_LINE_RE = re.compile(
    r"^(\d{1,2}|[A-Ha-h])[\.\)\:\-\s]\s*\S",
    re.UNICODE,
)
# Linha começando com dígito + palavra (OCR às vezes come "1. " do "1. An API...")
_OPTION_FRAG_RE = re.compile(r"^\d{1,2}\s+[A-Za-z]", re.UNICODE)
_JUNK_LINE_RE = re.compile(r"https?://|instructure\.com/courses/|^\s*[x×]\s+", re.I)
_URL_FRAGMENT_RE = re.compile(
    r"\.com/|module\s+item\s+id|/courses/|/assignments/|\?module\b|ture\.com|instructure|hitps?://|©\s*Bs\s+aer",
    re.I,
)
_BREADCRUMB_ASSIGNMENTS_RE = re.compile(
    r"^\s*[\d\-]+\s*>\s*Assignments|>\s*Assignments\s*»|ACDv2EN|LTI13-\d+",
    re.I,
)
# Opções em LMS/Canvas muitas vezes não vêm numeradas no OCR (radio buttons).
_UNNUM_MCQ_START_RE = re.compile(
    r"^(An API\b|A proxy\b|A service\b|Export\b|Use\b|Create\b|Select\b|Choose\b|Enable\b|Disable\b)",
    re.I,
)
_DATE_TASKBAR_RE = re.compile(r"\b\d{1,2}/\d{1,2}/\d{2,4}\b")
_LMS_MENU_ONLY_RE = re.compile(
    r"^(KEYBOARD\s+NAVIGATION|Dashboard|Calendar|Inbox|Account|Modules?|Grades?|"
    r"Discussions?|Announcements?|Courses?|Home|Help)\s*$",
    re.I,
)
_OPTION_BLOCK_SPLIT_RE = re.compile(
    # Suporta 1..10 (às vezes OCR traz 10.), e tolera ruído tipo "o 1."
    r"(?m)^\s*[oO,]*\s*([1-9]\d?)\.\s+(.+?)(?=^\s*[oO,]*\s*[1-9]\d?\.\s+|\Z)",
    re.DOTALL,
)
_MAX_OPTIONS_PER_QUESTION = 10


def _norm_for_match(s: str) -> str:
    s = (s or "").lower().strip()
    s = re.sub(r"^\s*([a-h]|\d{1,2})[\.\)\:\-]\s*", "", s, flags=re.I)
    s = re.sub(r"\s+", " ", s)
    s = re.sub(r"[^a-z0-9áàâãéêíóôõúç\s]", "", s, flags=re.I)
    return s.strip()


def _best_fuzzy_match_index(target: str, candidates: list[str]) -> tuple[int, float]:
    """Retorna (idx, score 0..1) do melhor match por similaridade (stdlib)."""
    import difflib

    t = _norm_for_match(target)
    if not t or not candidates:
        return -1, 0.0
    best_i, best_s = -1, 0.0
    for i, c in enumerate(candidates):
        cs = _norm_for_match(c)
        if not cs:
            continue
        s = difflib.SequenceMatcher(a=t, b=cs).ratio()
        if s > best_s:
            best_i, best_s = i, s
    return best_i, best_s


def _align_llm_item_to_ocr(it: dict, ocr_opts: list[str]) -> None:
    """
    Alinha opções do LLM com as opções do OCR e garante que 'suggested' aponte para o OCR.
    Adiciona campos:
      - suggested_ocr: letra A.. (índice no OCR)
      - suggested_text: texto da opção OCR correspondente
      - options_ocr: lista de opções OCR (normalizada)
    """
    if not isinstance(it, dict):
        return
    ocr_opts = [str(o).strip() for o in (ocr_opts or []) if str(o).strip()]
    if not ocr_opts:
        return

    it["options_ocr"] = ocr_opts

    sug = (it.get("suggested") or "").strip()
    if not sug:
        return

    # Primeiro, tentar mapear sugestão diretamente para OCR (A/B/1/2 etc)
    label, full = InvisibleScreenCapture._option_text_for_suggestion(sug, ocr_opts)
    if full:
        it["suggested_ocr"] = label.upper() if label else sug.upper()
        it["suggested_text"] = full
        return

    # Se não casou, tentar fuzzy-match usando as opções que o LLM retornou
    llm_opts = it.get("options") if isinstance(it.get("options"), list) else []
    llm_opts = [str(o).strip() for o in llm_opts if str(o).strip()]
    if not llm_opts:
        return

    # Descobrir qual opção o LLM “quis dizer”
    _lab, llm_full = InvisibleScreenCapture._option_text_for_suggestion(sug, llm_opts)
    if not llm_full:
        return

    idx, score = _best_fuzzy_match_index(llm_full, ocr_opts)
    if idx >= 0 and score >= 0.72:
        it["suggested_ocr"] = chr(65 + idx)
        it["suggested_text"] = ocr_opts[idx]
        if not (it.get("note") or "").strip():
            it["note"] = f"Realinhado por similaridade (score={score:.2f})."


def _env_flag(name: str) -> bool:
    return os.environ.get(name, "").strip().lower() in ("1", "true", "yes", "on")


def _env_flag_default(name: str, default: bool) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    return raw.strip().lower() in ("1", "true", "yes", "on")


def _parse_crop_frac(env_name: str, default: str) -> float:
    try:
        v = float(os.environ.get(env_name, default))
        return max(0.0, min(0.48, v))
    except ValueError:
        return float(default)


def _crop_image_bgr(img: np.ndarray, *, is_full_screen: bool) -> np.ndarray:
    """Recorta bordas antes do OCR: sidebar e barra de tarefas (env)."""
    h, w = img.shape[:2]
    if h < 80 or w < 80:
        return img
    if is_full_screen:
        lf = _parse_crop_frac("OCR_CROP_LEFT_FRAC", "0.18")
        rf = _parse_crop_frac("OCR_CROP_RIGHT_FRAC", "0.02")
        bf = _parse_crop_frac("OCR_CROP_BOTTOM_FRAC", "0.11")
        tf = _parse_crop_frac("OCR_CROP_TOP_FRAC", "0")
    else:
        lf = _parse_crop_frac("OCR_CROP_WINDOW_LEFT_FRAC", "0")
        rf = _parse_crop_frac("OCR_CROP_WINDOW_RIGHT_FRAC", "0")
        bf = _parse_crop_frac("OCR_CROP_WINDOW_BOTTOM_FRAC", "0")
        tf = _parse_crop_frac("OCR_CROP_WINDOW_TOP_FRAC", "0")
    x0, x1 = int(w * lf), int(w * (1 - rf))
    y0, y1 = int(h * tf), int(h * (1 - bf))
    if x1 <= x0 + 80 or y1 <= y0 + 80:
        return img
    return img[y0:y1, x0:x1].copy()


def _is_junk_continuation_line(s: str) -> bool:
    sl = s.lower()
    if _DATE_TASKBAR_RE.search(s):
        return True
    if "pesquisar" in sl and len(s) < 100:
        return True
    if "ensolarado" in sl or "pred " in sl or "°c" in s or "ºc" in sl:
        return True
    if re.search(r"\b\d{1,2}\s*°\s*c\b", s, re.I):
        return True
    if re.fullmatch(r"([a-zÀ-ÿ]\s+){4,}[a-zÀ-ÿ]?", sl):
        return True
    return False


def _line_looks_like_option_start(s: str) -> bool:
    s = s.strip()
    if len(s) < 4:
        return False
    if _OPTION_LINE_RE.match(s):
        return True
    if _OPTION_FRAG_RE.match(s):
        return True
    return False


def _split_options_from_block(block: str, max_opts: int = _MAX_OPTIONS_PER_QUESTION) -> list[str]:
    block = (block or "").strip()
    if not block:
        return []
    opts: list[str] = []
    for m in _OPTION_BLOCK_SPLIT_RE.finditer(block):
        num, body = m.group(1), m.group(2)
        body = re.sub(r"\s+", " ", body.replace("\n", " ")).strip()
        body = re.sub(r"^Help\s+", "", body, flags=re.I)
        if _is_junk_continuation_line(body) or len(body) < 6:
            continue
        opts.append(f"{num}. {body}")
        if len(opts) >= max_opts:
            break
    if len(opts) >= 2:
        return opts
    glued = re.sub(r"\s+", " ", block)
    if re.search(r"\b[1-8]\.\s+[A-Za-z]", glued):
        parts = re.split(r"(?=\b[1-8]\.\s+)", glued)
        out: list[str] = []
        for p in parts:
            p = p.strip()
            if re.match(r"^[1-8]\.\s+\S", p) and len(p) > 15 and not _is_junk_continuation_line(p):
                out.append(p)
        if len(out) >= 2:
            return out[:max_opts]
    return []


def _looks_like_url_line(s: str) -> bool:
    if not s:
        return False
    if _URL_FRAGMENT_RE.search(s):
        return True
    if "?" in s and re.search(r"[=&]\s*\d+|item\s+id\s*=", s, re.I):
        return True
    return False


def _is_continuation_mcq_line(c: str) -> bool:
    c = c.strip()
    if len(c) < 2:
        return False
    if c[0].islower():
        return True
    cl = c.lower()
    if cl.startswith(("that ", "and ", "which ", "style", "of the ", "to the ")):
        return True
    if len(c) <= 36 and "?" not in c and "API" not in c:
        return True
    return False


def _split_unnumbered_mcq_lines(block_lines: list[str], max_opts: int = _MAX_OPTIONS_PER_QUESTION) -> list[str]:
    """Opções sem '1.' no OCR (ex.: linhas 'An API...', 'A proxy...')."""
    opts: list[str] = []
    for raw in block_lines:
        c = raw.strip()
        if not c or re.fullmatch(r"[<>\s©]+", c):
            continue
        if _looks_like_url_line(c):
            continue
        if _BREADCRUMB_ASSIGNMENTS_RE.search(c) and "Which" not in c and "What" not in c:
            continue
        if _UNNUM_MCQ_START_RE.match(c):
            opts.append(c)
            continue
        if re.match(r"^(An|A)\s+\w+", c, re.I) and len(c) > 22:
            opts.append(c)
            continue
        if opts and _is_continuation_mcq_line(c):
            opts[-1] = (opts[-1] + " " + c).strip()
            continue
        # Heurística mais geral: linha “frase” com inicial maiúscula pode ser opção.
        if not opts:
            # só começa a coletar se a linha estiver com cara de alternativa
            if len(c) > 18 and c[0].isupper() and not _is_question_text(c):
                if not _looks_like_url_line(c):
                    opts.append(c)
            continue
        if len(c) > 12 and c[0].isupper() and not _is_question_text(c) and not _looks_like_url_line(c):
            opts.append(c)
    return opts[:max_opts]


def _pick_best_options(
    split_opts: list[str],
    fallback_opts: list[str],
    unnum_opts: list[str],
) -> list[str]:
    def score(lst: list[str]) -> tuple[int, int]:
        if not lst:
            return (-1, 0)
        n = len(lst)
        total = sum(len(x) for x in lst)
        return (n, -abs(total - 400))

    candidates = [split_opts, fallback_opts, unnum_opts]
    best = max(candidates, key=score)
    if score(best)[0] < 0:
        return []
    return best


def _build_llm_compact_block(questions: list[str], options_by_question: list[list[str]]) -> str:
    parts: list[str] = []
    for qi, q in enumerate(questions):
        parts.append(q.strip())
        if qi < len(options_by_question):
            for oi, opt in enumerate(options_by_question[qi], start=1):
                parts.append(f"{oi}. {opt.strip()}")
        parts.append("")
    return "\n".join(parts).strip()


def _collect_options_by_geometry(
    lines: list[dict], q_x0: int, max_opts: int = _MAX_OPTIONS_PER_QUESTION,
) -> list[str]:
    """Detecta opções por indentação (x0) relativa à pergunta + padrões de início de opção."""
    if not lines:
        return []
    opts: list[str] = []
    for ln in lines:
        s = ln["text"].strip()
        if not s or _is_junk_continuation_line(s):
            continue
        if _line_looks_like_option_start(s):
            opts.append(s)
            if len(opts) >= max_opts:
                break
            continue
        if ln["x0"] > q_x0 + 15 and len(s) > 20 and not _looks_like_url_line(s):
            if opts and not _line_looks_like_option_start(s):
                prev = opts[-1]
                if not prev.rstrip().endswith((".", "!", "?")):
                    opts[-1] = (prev + " " + s).strip()
            elif s[0].isupper() and len(s) > 30:
                opts.append(s)
    return opts[:max_opts]


def _pick_best_options_extended(
    split_opts: list[str],
    fallback_opts: list[str],
    unnum_opts: list[str],
    geo_opts: list[str],
) -> list[str]:
    """Escolhe o melhor conjunto de opções entre 4 estratégias de extração."""
    def score(lst: list[str]) -> tuple[int, int]:
        if not lst:
            return (-1, 0)
        n = len(lst)
        total = sum(len(x) for x in lst)
        return (n, -abs(total - 400))

    candidates = [split_opts, fallback_opts, unnum_opts, geo_opts]
    best = max(candidates, key=score)
    if score(best)[0] < 0:
        return []
    return best


_Q_START_RE = re.compile(
    r"^(\d+\.\s*)?(Qual|Como|Que|Q:|Pergunta:|Which|What|Select|Choose)\b",
    re.I,
)
_WH_AUX_RE = re.compile(
    r"^(\d+\.\s*)?(How|When|Where|Why)\s+"
    r"(do|does|did|is|are|can|could|should|would|will|must|many|much|to)\b",
    re.I,
)


def _is_question_text(s: str) -> bool:
    """Retorna True se o texto parece ser o início de uma pergunta."""
    s = s.strip()
    if _looks_like_url_line(s):
        return False
    if _BREADCRUMB_ASSIGNMENTS_RE.search(s) and "Which" not in s and "What" not in s:
        return False
    if s and s[0].islower() and "?" not in s:
        if not re.match(
            r"^(which|what|how|when|where|why|select|choose|qual|como|quando|onde)\b",
            s, re.I,
        ):
            return False
    if _Q_START_RE.match(s):
        return True
    if _WH_AUX_RE.match(s):
        return True
    if "?" in s and not _URL_FRAGMENT_RE.search(s):
        if len(s) < 500 and re.search(
            r"\b(describe|best|correct|true|false|select|choose|suggests|option|phrase)\b",
            s, re.I,
        ):
            return True
    return False


def _parse_questions_structured(
    lines: list[dict],
) -> tuple[list[str], list[list[str]]]:
    """
    Parser de perguntas/opções usando padrões de texto + geometria (x0/y0).
    Agrupa continuações de enunciado por proximidade de x0 e mescla linhas.
    """
    if not lines:
        return [], []

    questions: list[str] = []
    options_by_question: list[list[str]] = []

    i = 0
    while i < len(lines):
        ln = lines[i]
        if _is_question_text(ln["text"]):
            q_text = _clean_question_line(ln["text"])
            q_x0 = ln["x0"]

            j = i + 1
            while j < len(lines):
                nxt = lines[j]
                ns = nxt["text"].strip()
                if _is_question_text(ns):
                    break
                if _line_looks_like_option_start(ns):
                    break
                starts_lower = ns and (
                    ns[0].islower()
                    or ns.startswith(("that ", "and ", "which ", "of the ", "to the ", "Select ", "style"))
                )
                starts_upper_cont = (
                    ns and ns[0].isupper()
                    and not _line_looks_like_option_start(ns)
                    and not _is_question_text(ns)
                )
                is_cont = (
                    abs(nxt["x0"] - q_x0) < 60
                    and ns
                    and (starts_lower or starts_upper_cont)
                    and len(q_text) < 500
                    and not _is_junk_continuation_line(ns)
                )
                if is_cont:
                    q_text = q_text + " " + ns
                    j += 1
                    continue
                break

            questions.append(q_text if q_text else ln["text"])

            block_text_lines: list[str] = []
            block_structured: list[dict] = []
            while j < len(lines):
                candidate = lines[j]
                if _is_question_text(candidate["text"]):
                    break
                block_text_lines.append(candidate["text"])
                block_structured.append(candidate)
                j += 1

            block_text = "\n".join(block_text_lines)
            split_opts = _split_options_from_block(block_text, _MAX_OPTIONS_PER_QUESTION)
            fb_opts = _fallback_collect_options(block_text_lines, _MAX_OPTIONS_PER_QUESTION)
            un_opts = _split_unnumbered_mcq_lines(block_text_lines, _MAX_OPTIONS_PER_QUESTION)
            geo_opts = _collect_options_by_geometry(block_structured, q_x0)

            opts = _pick_best_options_extended(split_opts, fb_opts, un_opts, geo_opts)
            options_by_question.append(opts)
            i = j
            continue
        i += 1

    return questions, options_by_question


def _summarize_items(
    questions: list[str], options_by_question: list[list[str]],
) -> list[dict]:
    """Gera resumos (full + short) para cada pergunta, remove duplicatas de opções."""
    items: list[dict] = []
    for i, q in enumerate(questions):
        opts = options_by_question[i] if i < len(options_by_question) else []
        seen: set[str] = set()
        unique_opts: list[str] = []
        for opt in opts:
            normalized = re.sub(r"\s+", " ", opt.strip().lower())
            if normalized not in seen:
                seen.add(normalized)
                unique_opts.append(opt)

        q_short = (q[:200] + "...") if len(q) > 200 else q
        opts_short = [
            (opt[:120] + "...") if len(opt) > 120 else opt
            for opt in unique_opts
        ]
        items.append({
            "question": q,
            "question_short": q_short,
            "options": unique_opts,
            "options_short": opts_short,
            "complete": len(unique_opts) >= 2,
        })
    return items


def _fallback_collect_options(block_lines: list[str], max_opts: int = _MAX_OPTIONS_PER_QUESTION) -> list[str]:
    opts: list[str] = []
    for candidate in block_lines:
        c = candidate.strip()
        if not c or _is_junk_continuation_line(c):
            continue
        if _line_looks_like_option_start(c):
            opts.append(c)
            if len(opts) >= max_opts:
                break
            continue
        cont = (c[0].islower() if c else False) or c.startswith(
            ("that ", "and ", "with ", "the ", "to ", "of ", "on ", "in ")
        )
        # OCR por wrap pode quebrar a alternativa e iniciar a continuação com letra maiúscula.
        # Só fazemos merge quando a linha atual não parece um começo de opção e não parece
        # um novo enunciado.
        cont_upper_wrap = bool(
            c
            and c[0].isupper()
            and not _line_looks_like_option_start(c)
            and not _is_question_text(c)
        )
        cont = cont or cont_upper_wrap
        if (
            opts
            and len(c) > 8
            and "?" not in c
            and not _is_junk_continuation_line(c)
            and cont
        ):
            prev = opts[-1]
            if not prev.rstrip().endswith((".", "!", "?")):
                opts[-1] = (prev + " " + c).strip()
    return opts[:max_opts]

# Win10 2004+: janela visível no seu monitor, mas não aparece na maioria dos compartilhamentos de tela
_WDA_EXCLUDEFROMCAPTURE = 0x00000011
_WDA_NONE = 0x00000000
# GetAncestor: https://learn.microsoft.com/windows/win32/api/winuser/nf-winuser-getancestor
_GA_ROOT = 2
_GWL_EXSTYLE = -20
_WS_EX_LAYERED = 0x00080000


def _set_window_display_affinity(hwnd: int, affinity: int) -> tuple[bool, str]:
    if os.name != "nt" or hwnd <= 0:
        return False, "Não-Windows ou HWND inválido."
    try:
        user32 = ctypes.windll.user32
        user32.SetWindowDisplayAffinity.argtypes = [ctypes.c_void_p, ctypes.c_uint]
        user32.SetWindowDisplayAffinity.restype = ctypes.c_int
        ok = bool(user32.SetWindowDisplayAffinity(hwnd, affinity))
        if ok:
            return True, "OK"
        # Nem sempre há GetLastError útil aqui, mas ajuda quando existe.
        try:
            kernel32 = ctypes.windll.kernel32
            kernel32.GetLastError.restype = ctypes.c_uint
            err = int(kernel32.GetLastError())
        except Exception:
            err = 0
        return False, f"Falhou (GetLastError={err})"
    except Exception:
        return False, "Exceção ao chamar SetWindowDisplayAffinity."


def _hwnd_exclude_from_screen_capture(hwnd: int) -> tuple[bool, str]:
    return _set_window_display_affinity(hwnd, _WDA_EXCLUDEFROMCAPTURE)


def _collect_hwnd_candidates(tk_hwnd: int) -> list[int]:
    """
    No Tk/Win32, winfo_id() às vezes é filho; SetWindowDisplayAffinity no HWND errado não surte efeito.
    Tenta o HWND do Tk, o ancestral ROOT e o pai — ordem favorece ROOT primeiro na aplicação.
    """
    out: list[int] = []
    if tk_hwnd <= 0 or os.name != "nt":
        return [tk_hwnd] if tk_hwnd > 0 else []

    def add(h: int) -> None:
        if h > 0 and h not in out:
            out.append(h)

    try:
        user32 = ctypes.windll.user32
        user32.GetAncestor.argtypes = [ctypes.c_void_p, ctypes.c_uint]
        user32.GetAncestor.restype = ctypes.c_void_p
        user32.GetParent.argtypes = [ctypes.c_void_p]
        user32.GetParent.restype = ctypes.c_void_p

        root = int(user32.GetAncestor(tk_hwnd, _GA_ROOT) or 0)
        parent = int(user32.GetParent(tk_hwnd) or 0)
        # ROOT costuma ser o top-level que o DWM compõe na captura de desktop
        add(root)
        add(tk_hwnd)
        add(parent)
    except Exception:
        add(tk_hwnd)
    return out


def _ensure_ws_ex_layered(hwnd: int) -> tuple[bool, str]:
    """Opcional: alguns fluxos combinam layered + afinidade (não é garantia universal)."""
    if hwnd <= 0 or os.name != "nt":
        return False, "HWND inválido ou não-Windows."
    try:
        user32 = ctypes.windll.user32
        if hasattr(user32, "GetWindowLongPtrW"):
            get_long = user32.GetWindowLongPtrW
            set_long = user32.SetWindowLongPtrW
            get_long.argtypes = [ctypes.c_void_p, ctypes.c_int]
            get_long.restype = ctypes.c_void_p
            set_long.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_void_p]
            set_long.restype = ctypes.c_void_p
            style = int(get_long(hwnd, _GWL_EXSTYLE) or 0)
            new_style = style | _WS_EX_LAYERED
            if new_style != style:
                set_long(hwnd, _GWL_EXSTYLE, ctypes.c_void_p(new_style))
        else:
            get_long = user32.GetWindowLongW
            set_long = user32.SetWindowLongW
            get_long.argtypes = [ctypes.c_void_p, ctypes.c_int]
            get_long.restype = ctypes.c_long
            set_long.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_long]
            set_long.restype = ctypes.c_long
            style = int(get_long(hwnd, _GWL_EXSTYLE))
            new_style = style | _WS_EX_LAYERED
            if new_style != style:
                set_long(hwnd, _GWL_EXSTYLE, new_style)
        return True, "OK"
    except Exception as e:
        return False, str(e)


def _apply_exclude_from_capture_to_tk_root(root: tk.Misc) -> tuple[bool, str]:
    """Aplica WDA_EXCLUDEFROMCAPTURE no(s) HWND(s) plausível(is) do root Tk."""
    root.update_idletasks()
    tk_hwnd = int(root.winfo_id())
    candidates = _collect_hwnd_candidates(tk_hwnd)
    if not candidates:
        return False, "Nenhum HWND candidato."

    last_detail = ""
    for h in candidates:
        ok, msg = _hwnd_exclude_from_screen_capture(h)
        if ok:
            return True, f"HWND 0x{h:x} | {msg}"
        last_detail = f"0x{h:x}: {msg}"

    if _env_flag("EXCLUDE_TRY_LAYERED") and candidates:
        h0 = candidates[0]
        _ok_l, _m = _ensure_ws_ex_layered(h0)
        for h in candidates:
            ok, msg = _hwnd_exclude_from_screen_capture(h)
            if ok:
                return True, f"HWND 0x{h:x} (após WS_EX_LAYERED) | {msg}"
        last_detail = f"{last_detail}; layered({_m}): ainda falhou"

    return False, last_detail


def _preprocess_for_ocr(img_bgr: np.ndarray) -> np.ndarray:
    """Binarizacao Otsu + CLAHE + sharpening para OCR preciso em paginas web."""
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape[:2]
    fast = _env_flag("OCR_FAST")
    target_w = 1400 if fast else 2000
    cap = 1.35 if fast else 2.5
    if w < target_w:
        scale = min(cap, target_w / max(w, 1))
        gray = cv2.resize(gray, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
    _, gray = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    # Card escuro (Canvas) pode ficar “preto demais” após Otsu; inverte para ajudar o Tesseract.
    try:
        if float(np.mean(gray)) < 80:
            gray = 255 - gray
    except Exception:
        pass
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    gray = clahe.apply(gray)
    if fast:
        return gray
    blur = cv2.GaussianBlur(gray, (0, 0), 1.0)
    return cv2.addWeighted(gray, 1.5, blur, -0.5, 0)


def _crop_quiz_card_if_found(img_bgr: np.ndarray) -> np.ndarray:
    """
    Tenta recortar o card do quiz (Canvas/LMS) para reduzir ruído de sidebar/header.
    Heurística: encontrar o maior retângulo escuro na metade direita (onde fica o conteúdo).
    """
    try:
        h, w = img_bgr.shape[:2]
        if h < 300 or w < 400:
            return img_bgr
        # focar na área principal (ignorar sidebar esquerda)
        x0 = int(w * 0.22)
        roi = img_bgr[:, x0:].copy()
        g = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        # máscara de pixels escuros (card)
        _, m = cv2.threshold(g, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        # limpar ruído e unir regiões
        k = cv2.getStructuringElement(cv2.MORPH_RECT, (9, 9))
        m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, k, iterations=2)
        cnts, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not cnts:
            return img_bgr
        best = None
        best_area = 0
        for c in cnts:
            x, y, cw, ch = cv2.boundingRect(c)
            area = cw * ch
            if area < best_area:
                continue
            # card típico é grande e “retangular”
            if cw < int((w - x0) * 0.35) or ch < int(h * 0.25):
                continue
            ar = cw / max(ch, 1)
            if ar < 0.7 or ar > 4.5:
                continue
            best = (x, y, cw, ch)
            best_area = area
        if not best:
            return img_bgr
        x, y, cw, ch = best
        pad = 18
        xA = max(0, x0 + x - pad)
        yA = max(0, y - pad)
        xB = min(w, x0 + x + cw + pad)
        yB = min(h, y + ch + pad)
        cropped = img_bgr[yA:yB, xA:xB].copy()
        # evitar recorte absurdo
        if cropped.shape[0] < 220 or cropped.shape[1] < 300:
            return img_bgr
        return cropped
    except Exception:
        return img_bgr


def _ocr_best_of_variants(img_bgr: np.ndarray, lang: str, config: str) -> tuple[list[dict], str]:
    """
    Tenta variantes de OCR (recorte + preprocess diferentes) e escolhe a melhor
    pela quantidade/qualidade de linhas.
    """
    variants: list[tuple[str, np.ndarray]] = []

    base = _crop_quiz_card_if_found(img_bgr)
    variants.append(("crop+otsu", _preprocess_for_ocr(base)))

    # Variante: adaptive threshold (melhor para fundo irregular/anti-alias)
    try:
        g = cv2.cvtColor(base, cv2.COLOR_BGR2GRAY)
        g = cv2.GaussianBlur(g, (3, 3), 0)
        ad = cv2.adaptiveThreshold(
            g, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 41, 7
        )
        if float(np.mean(ad)) < 80:
            ad = 255 - ad
        variants.append(("crop+adaptive", ad))
    except Exception:
        pass

    best_lines: list[dict] = []
    best_raw = ""
    best_score = -1.0
    for _name, proc in variants:
        lines, raw = _ocr_to_structured_lines(proc, lang, config)
        filtered = _filter_structured_lines(lines)
        # score: mais linhas + mais palavras “úteis”
        wc = sum(int(ln.get("word_count") or 0) for ln in filtered)
        score = len(filtered) * 3 + wc
        if score > best_score:
            best_score = score
            best_lines = filtered
            best_raw = raw
    return best_lines, best_raw


_MIN_WORD_CONF = int(os.environ.get("OCR_MIN_WORD_CONF", "15"))


def _ocr_to_structured_lines(
    proc_img, lang: str, config: str,
) -> tuple[list[dict], str]:
    """
    OCR estruturado via image_to_data: retorna bounding boxes + confiança por palavra,
    agrupadas em linhas lógicas. Cada linha é um dict com text, x0, y0, y1, avg_conf, word_count.
    Retorna (linhas, texto bruto).
    """
    try:
        data = pytesseract.image_to_data(
            proc_img, lang=lang, config=config,
            output_type=pytesseract.Output.DICT,
        )
    except Exception:
        return _ocr_fallback_lines(proc_img, lang, config)

    n = len(data.get("text", []))
    if n == 0:
        return _ocr_fallback_lines(proc_img, lang, config)

    line_groups: dict[tuple[int, int, int], list[dict]] = {}
    for i in range(n):
        txt = (data["text"][i] or "").strip()
        if not txt:
            continue
        raw_conf = str(data["conf"][i]).lstrip("-")
        conf = int(raw_conf) if raw_conf.isdigit() else -1
        if 0 <= conf < _MIN_WORD_CONF:
            continue
        key = (int(data["block_num"][i]), int(data["par_num"][i]), int(data["line_num"][i]))
        if key not in line_groups:
            line_groups[key] = []
        line_groups[key].append({
            "text": txt,
            "conf": max(conf, 0),
            "left": int(data["left"][i]),
            "top": int(data["top"][i]),
            "width": int(data["width"][i]),
            "height": int(data["height"][i]),
        })

    raw_lines: list[dict] = []
    for key in sorted(line_groups.keys()):
        words = line_groups[key]
        words.sort(key=lambda w: w["left"])
        text = " ".join(w["text"] for w in words)
        x0 = words[0]["left"]
        y0 = min(w["top"] for w in words)
        y1 = max(w["top"] + w["height"] for w in words)
        confs = [w["conf"] for w in words if w["conf"] > 0]
        avg_conf = sum(confs) / len(confs) if confs else 0.0
        raw_lines.append({
            "text": text, "x0": x0, "y0": y0, "y1": y1,
            "avg_conf": avg_conf, "word_count": len(words),
        })

    raw_lines.sort(key=lambda ln: (ln["y0"], ln["x0"]))

    lines: list[dict] = []
    for ln in raw_lines:
        if lines:
            prev = lines[-1]
            overlap_top = max(prev["y0"], ln["y0"])
            overlap_bot = min(prev["y1"], ln["y1"])
            prev_h = prev["y1"] - prev["y0"]
            ln_h = ln["y1"] - ln["y0"]
            min_h = min(prev_h, ln_h) if min(prev_h, ln_h) > 0 else 1
            if (overlap_bot - overlap_top) / min_h > 0.5:
                prev["text"] = prev["text"] + " " + ln["text"]
                prev["y1"] = max(prev["y1"], ln["y1"])
                prev["word_count"] += ln["word_count"]
                prev["avg_conf"] = (prev["avg_conf"] + ln["avg_conf"]) / 2
                continue
        lines.append(dict(ln))

    raw_text = "\n".join(ln["text"] for ln in lines)
    return lines, raw_text


def _ocr_fallback_lines(proc_img, lang: str, config: str) -> tuple[list[dict], str]:
    """Fallback para image_to_string quando image_to_data falha."""
    try:
        raw = pytesseract.image_to_string(proc_img, lang=lang, config=config)
    except Exception:
        raw = pytesseract.image_to_string(proc_img, lang="eng", config=config)
    lines: list[dict] = []
    for i, line_text in enumerate(raw.split("\n")):
        s = line_text.strip()
        if s:
            lines.append({
                "text": s, "x0": 0, "y0": i * 20, "y1": (i + 1) * 20,
                "avg_conf": 50.0, "word_count": len(s.split()),
            })
    return lines, raw


def _is_junk_text(s: str) -> bool:
    """Testa se a string casa com padrões de lixo (URLs, taskbar, menu, etc.)."""
    if len(s) < 2:
        return True
    if _JUNK_LINE_RE.search(s):
        return True
    if s.count("|") >= 4 and len(s) > 40:
        return True
    if re.match(r"^[€$£\-\+\s\d\.\:]+$", s):
        return True
    if _DATE_TASKBAR_RE.search(s) and len(s) < 120:
        return True
    if "pesquisar" in s.lower() and len(s) < 90:
        return True
    if re.search(r"°\s*c|ºc|ensolarado|pred\s+ens", s, re.I):
        return True
    if _LMS_MENU_ONLY_RE.match(s):
        return True
    if re.search(r"Lucid\s*\(\s*Whiteboard", s, re.I) and len(s) < 100:
        return True
    if re.match(
        r"^(Due|Points|Submitting|No\s+Due\s+Date|external\s+tool)\b", s, re.I
    ) and len(s) < 80:
        return True
    if _URL_FRAGMENT_RE.search(s) and len(s) < 220:
        return True
    if _BREADCRUMB_ASSIGNMENTS_RE.search(s) and "Which" not in s and "What" not in s and len(s) < 200:
        return True
    if re.match(r"^Module\s+\d+\s+Knowledge\s+Check\s*$", s, re.I):
        return True
    return False


def _filter_structured_lines(lines: list[dict]) -> list[dict]:
    """Filtra linhas estruturadas removendo lixo por texto e baixa confiança."""
    out: list[dict] = []
    for ln in lines:
        if ln["avg_conf"] < 10 and ln["avg_conf"] > 0 and ln["word_count"] < 5:
            continue
        if _is_junk_text(ln["text"].strip()):
            continue
        out.append(ln)
    return out


def _filter_ocr_lines(lines: list[str]) -> list[str]:
    """Remove abas, URLs, LMS sidebar e padrões de barra de tarefas."""
    out: list[str] = []
    for line in lines:
        s = line.strip()
        if len(s) < 2:
            continue
        if _JUNK_LINE_RE.search(s):
            continue
        if s.count("|") >= 4 and len(s) > 40:
            continue
        if re.match(r"^[€$£\-\+\s\d\.\:]+$", s):
            continue
        if _DATE_TASKBAR_RE.search(s) and len(s) < 120:
            continue
        if "pesquisar" in s.lower() and len(s) < 90:
            continue
        if re.search(r"°\s*c|ºc|ensolarado|pred\s+ens", s, re.I):
            continue
        if _LMS_MENU_ONLY_RE.match(s):
            continue
        if re.search(r"Lucid\s*\(\s*Whiteboard", s, re.I) and len(s) < 100:
            continue
        if re.match(
            r"^(Due|Points|Submitting|No\s+Due\s+Date|external\s+tool)\b", s, re.I
        ) and len(s) < 80:
            continue
        if _URL_FRAGMENT_RE.search(s) and len(s) < 220:
            continue
        if _BREADCRUMB_ASSIGNMENTS_RE.search(s) and "Which" not in s and "What" not in s and len(s) < 200:
            continue
        if re.match(r"^Module\s+6\s+Knowledge\s+Check\s*$", s, re.I):
            continue
        out.append(s)
    return out


def _clean_question_line(line: str) -> str:
    """Remove lixo OCR antes da pergunta (ex.: fragmento de sidebar + '1. Which...')."""
    m = re.search(r"(\d+\.\s*)?(Which|What|How|When|Where|Why|Select|Choose|Qual|Como)\b", line, re.I)
    if m:
        return line[m.start() :].strip()
    return line.strip()


def _offline_websocket_suggestion(
    question: str, options: list[str]
) -> tuple[str | None, str | None, str]:
    """WebSocket = tempo real / chat; não batch nem relatório estático."""
    q = (question or "").lower()
    opts = [str(o).strip() for o in (options or []) if str(o).strip()]
    if not opts or ("websocket" not in q and "web socket" not in q):
        return None, None, ""
    if not any(k in q for k in ("suggest", "option", "which", "what", "best", "use", "scenario", "case")):
        return None, None, ""

    def letter_for_index(i: int) -> str:
        return chr(65 + i) if i < 26 else "?"

    for i, op in enumerate(opts):
        lo = op.lower()
        if ("real-time" in lo or "real time" in lo) and (
            "chat" in lo or "support" in lo or "customer" in lo
        ):
            return (
                letter_for_index(i),
                op,
                "WebSocket: comunicação bidirecional em tempo real (ex.: chat).",
            )
    for i, op in enumerate(opts):
        lo = op.lower()
        if "batch" in lo or ("static" in lo and "report" in lo):
            continue
        if "real-time" in lo or "real time" in lo:
            return (
                letter_for_index(i),
                op,
                "Cenário em tempo real combina com WebSocket.",
            )
        if "chat" in lo and "application" in lo:
            return (
                letter_for_index(i),
                op,
                "Chat ao vivo costuma usar WebSocket.",
            )
    return None, None, ""


def _offline_fallback_suggestion(question: str, options: list[str]) -> tuple[str | None, str | None, str]:
    """
    Fallback sem LLM: heurísticas mínimas por texto da pergunta + alternativas (sem rede).
    Retorna (letra A-D, texto da opção, nota curta) ou (None, None, "").
    """
    q = (question or "").lower()
    opts = [str(o).strip() for o in (options or []) if str(o).strip()]
    if not opts:
        return None, None, ""

    def letter_for_index(i: int) -> str:
        return chr(65 + i) if i < 26 else "?"

    ws = _offline_websocket_suggestion(question, opts)
    if ws[0]:
        return ws

    # Pergunta muito comum em cursos AWS/Canvas
    if ("restful" in q or "rest " in q) and "api" in q:
        if any(k in q for k in ("describe", "phrase", "which", "what", "best", "mean")):
            for i, op in enumerate(opts):
                lo = op.lower()
                if "representational state transfer" in lo:
                    return (
                        letter_for_index(i),
                        op,
                        "Padrão de livro: REST é o estilo arquitetural (não é só “usa HTTP”).",
                    )
                if "principles" in lo and "representational" in lo:
                    return (
                        letter_for_index(i),
                        op,
                        "Opção alinhada à definição de estilo REST.",
                    )

    return None, None, ""


_STOP_KEYWORD_OVERLAP = frozenset(
    {
        "the",
        "a",
        "an",
        "is",
        "are",
        "was",
        "were",
        "which",
        "what",
        "when",
        "where",
        "who",
        "how",
        "why",
        "does",
        "do",
        "did",
        "can",
        "could",
        "should",
        "would",
        "this",
        "that",
        "these",
        "those",
        "best",
        "phrase",
        "describes",
        "true",
        "false",
        "following",
        "select",
        "choose",
        "one",
        "most",
        "correct",
        "any",
        "all",
    }
)


def _infer_suggested_by_keywords(question: str, options: list[str]) -> str | None:
    """Último recurso: escolhe a alternativa com mais termos em comum com o enunciado (revisar)."""
    ql = (question or "").lower()
    if any(x in ql for x in (" not ", " except ", " least ", " incorrect ", " false ")):
        return None
    qwords = set(re.findall(r"[a-z][a-z0-9\-]{2,}", ql)) - _STOP_KEYWORD_OVERLAP
    if len(qwords) < 2:
        return None
    best_i, best_score = -1, -1
    for i, opt in enumerate(options):
        owords = set(re.findall(r"[a-z][a-z0-9\-]{2,}", opt.lower()))
        score = len(qwords & owords)
        if score > best_score:
            best_score, best_i = score, i
    if best_score >= 2 and 0 <= best_i < len(options):
        return chr(65 + best_i)
    return None


def enrich_parsed_items(parsed: dict | None) -> dict | None:
    """Preenche suggested vazio após o modelo; corrige erro típico IA em perguntas WebSocket."""
    if not parsed or not isinstance(parsed.get("items"), list):
        return parsed
    for it in parsed["items"]:
        if not isinstance(it, dict):
            continue
        q = (it.get("question") or "").strip()
        opts = [str(o).strip() for o in (it.get("options") or []) if str(o).strip()]
        if not q or not opts:
            continue
        sug = "" if it.get("suggested") is None else str(it.get("suggested")).strip()
        if not sug:
            letter, _txt, why = _offline_fallback_suggestion(q, opts)
            if letter:
                it["suggested"] = letter
                it["confidence"] = (it.get("confidence") or "").strip() or "media"
                if not (it.get("note") or "").strip():
                    it["note"] = why
            else:
                kw = _infer_suggested_by_keywords(q, opts)
                if kw:
                    it["suggested"] = kw
                    it["confidence"] = "baixa"
                    prev = (it.get("note") or "").strip()
                    extra = "Estimativa por palavras-chave no enunciado (revisar)."
                    it["note"] = f"{prev} {extra}".strip() if prev else extra
        w_letter, _w_txt, w_note = _offline_websocket_suggestion(q, opts)
        if w_letter:
            it["suggested"] = w_letter
            it["confidence"] = "alta"
            it["note"] = w_note
    return parsed


def align_llm_to_ocr(parsed: dict | None, options_by_question: list[list[str]]) -> dict | None:
    """Realinha itens do LLM com as opções detectadas no OCR (pós-processamento)."""
    if not parsed or not isinstance(parsed.get("items"), list):
        return parsed
    items = parsed.get("items") or []
    if not isinstance(items, list):
        return parsed
    for i, it in enumerate(items):
        ocr_opts = options_by_question[i] if i < len(options_by_question) else []
        _align_llm_item_to_ocr(it, ocr_opts)
    return parsed


def _all_offline_covers_every_question(
    questions: list[str], options_by_question: list[list[str]]
) -> bool:
    if not questions:
        return False
    for i, q in enumerate(questions):
        opts = options_by_question[i] if i < len(options_by_question) else []
        letter, _, _ = _offline_fallback_suggestion(q, opts)
        if not letter:
            return False
    return True


class InvisibleScreenCapture:
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("Invisible Capture")
        self.root.geometry("420x350+10+10")

        self.root.attributes("-alpha", 0.98)
        self.root.attributes("-topmost", True)
        # Por padrão a janela é "borderless" (não aparece na barra de tarefas em alguns setups).
        # Para depurar/usar no dia a dia: UI_BORDERLESS=0
        self.root.overrideredirect(_env_flag_default("UI_BORDERLESS", False))

        # mss no Windows usa estado por thread — não reutilizar instância fora da thread que captura
        self._ocr_busy = False
        self._llm_provider: AirLLMProvider | None = None

        self._web_store = CaptureStore(
            max_history=int(os.environ.get("WEB_HISTORY", "25")),
        )
        self._web_server = None
        self._web_port = int(os.environ.get("WEB_PORT", "8765"))
        self._web_url = f"http://127.0.0.1:{self._web_port}"
        if _env_flag("WEB_ENABLE"):
            try:
                self._web_server, self._web_port, self._web_url = start_web_server(
                    self._web_store, self._web_port,
                )
                print(f"Servidor web: {self._web_url}")
            except Exception as e:
                print(f"Falha ao iniciar servidor web: {e}")

        keyboard.add_hotkey("ctrl+shift+c", lambda: self.root.after(0, self.toggle_visibility))
        keyboard.add_hotkey("ctrl+shift+s", lambda: self.root.after(0, self.capture_screen))
        keyboard.add_hotkey("ctrl+shift+w", lambda: self.root.after(0, self.capture_active_window))

        self.root.protocol("WM_DELETE_WINDOW", self.on_closing)

        self.setup_ui()
        # Aplicar logo após criação e também quando a janela for "mapeada" (aparecer de fato).
        self.root.after(0, self._apply_exclude_from_capture)
        self.root.bind("<Map>", lambda _e: self.root.after(0, self._apply_exclude_from_capture))
        self.root.mainloop()

    def _apply_exclude_from_capture(self):
        ok, msg = _apply_exclude_from_capture_to_tk_root(self.root)
        if ok:
            self.status_label.config(
                text=f"Exclusão de captura: OK ({msg}) | Hotkeys: Ctrl+Shift+C, S, W"
            )
            return
        if os.name == "nt":
            self.status_label.config(
                text=(
                    "Aviso: exclusão de captura não aplicada em nenhum HWND candidato, "
                    "ou o capturador ignora WDA_EXCLUDEFROMCAPTURE (comum em Tela inteira). "
                    f"Detalhe: {msg} | Hotkeys: Ctrl+Shift+C, S, W"
                )
            )

    def setup_ui(self):
        main_frame = ttk.Frame(self.root)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        self.status_label = ttk.Label(
            main_frame,
            text="Pronto | Hotkeys: Ctrl+Shift+S (tela), Ctrl+Shift+W (janela), Ctrl+Shift+C (mostrar/ocultar)",
            font=("Arial", 9),
        )
        self.status_label.pack(pady=5)

        self.use_llm = tk.BooleanVar(value=True)
        self.chk_llm = ttk.Checkbutton(
            main_frame,
            text="Refinar com IA local após captura (AirLLM)",
            variable=self.use_llm,
        )
        self.chk_llm.pack(anchor=tk.W, pady=(0, 6))

        self.compact_quiz = tk.BooleanVar(value=_env_flag("UI_COMPACT_QUIZ"))
        self.chk_compact = ttk.Checkbutton(
            main_frame,
            text="Só pergunta e resposta (sem lista A–D nem OCR bruto)",
            variable=self.compact_quiz,
        )
        self.chk_compact.pack(anchor=tk.W, pady=(0, 4))

        text_frame = ttk.Frame(main_frame)
        text_frame.pack(fill=tk.BOTH, expand=True, pady=5)

        scrollbar = ttk.Scrollbar(text_frame)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

        self.text_area = tk.Text(
            text_frame,
            height=15,
            width=50,
            wrap=tk.WORD,
            font=("Consolas", 9),
            bg="#1e1e1e",
            fg="#ffffff",
        )
        self.text_area.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        self.text_area.config(yscrollcommand=scrollbar.set)
        scrollbar.config(command=self.text_area.yview)

        btn_frame = ttk.Frame(main_frame)
        btn_frame.pack(fill=tk.X, pady=5)

        self.btn_full = ttk.Button(btn_frame, text="Capturar Tela", command=self.capture_screen)
        self.btn_full.pack(side=tk.LEFT, padx=2)
        self.btn_win = ttk.Button(btn_frame, text="Capturar Janela Ativa", command=self.capture_active_window)
        self.btn_win.pack(side=tk.LEFT, padx=2)
        self.btn_warm = ttk.Button(btn_frame, text="Warmup AirLLM", command=self.warmup_llm)
        self.btn_warm.pack(side=tk.LEFT, padx=2)
        self.btn_clear = ttk.Button(btn_frame, text="Limpar", command=self.clear_text)
        self.btn_clear.pack(side=tk.LEFT, padx=2)
        self.btn_web = ttk.Button(btn_frame, text="Abrir Web", command=self._open_web_page)
        self.btn_web.pack(side=tk.LEFT, padx=2)
        self.btn_copy = ttk.Button(btn_frame, text="Copiar", command=self.copy_results)
        self.btn_copy.pack(side=tk.LEFT, padx=2)

        self._last_results: dict | None = None

    def _get_llm_provider(self) -> AirLLMProvider:
        if self._llm_provider is not None:
            return self._llm_provider
        self._llm_provider = AirLLMProvider(
            AirLLMConfig(model_id_or_path=os.environ.get("AIRLLM_MODEL", "").strip())
        )
        return self._llm_provider

    def warmup_llm(self):
        if self._ocr_busy:
            return
        self._set_busy(True)
        self.status_label.config(text="AirLLM: carregando modelo (warmup)...")

        def work():
            try:
                prov = self._get_llm_provider()
                prov.warmup()
                self.root.after(0, lambda: self.status_label.config(text="AirLLM: warmup concluído."))
            except Exception as e:
                self.root.after(0, lambda err=e: self.status_label.config(text=f"Erro no warmup AirLLM: {err}"))
            finally:
                self.root.after(0, lambda: self._set_busy(False))

        threading.Thread(target=work, daemon=True).start()

    def _open_web_page(self):
        if not self._web_server:
            try:
                self._web_server, self._web_port, self._web_url = start_web_server(
                    self._web_store, self._web_port,
                )
            except Exception as e:
                self.status_label.config(text=f"Erro ao iniciar servidor web: {e}")
                return
        # Sempre abrir no localhost neste PC (mais confiável que o IP da rede).
        webbrowser.open(f"http://127.0.0.1:{self._web_port}")

    def _set_busy(self, busy: bool):
        self._ocr_busy = busy
        state = "disabled" if busy else "normal"
        self.btn_full.configure(state=state)
        self.btn_win.configure(state=state)
        self.btn_warm.configure(state=state)
        self.chk_llm.configure(state=state)
        self.chk_compact.configure(state=state)

    def on_closing(self):
        try:
            keyboard.unhook_all()
        except Exception:
            pass
        self.root.destroy()

    def toggle_visibility(self):
        current_alpha = self.root.attributes("-alpha")
        if current_alpha == 0.0:
            self.root.attributes("-alpha", 0.95)
            self.root.geometry("420x350+10+10")
            self._apply_exclude_from_capture()
            self.status_label.config(
                text="Visível para você | Oculto no compartilhamento | Ctrl+Shift+C para esconder só da sua tela"
            )
        else:
            self.root.attributes("-alpha", 0.0)
            self.root.geometry("1x1+0+0")
            self.status_label.config(
                text="Minimizado (só para você) | Ctrl+Shift+C para restaurar | S/W continuam valendo"
            )

    def capture_screen(self):
        if self._ocr_busy:
            return
        self._set_busy(True)
        self.status_label.config(text="Capturando tela (OCR em segundo plano)...")
        use_llm = self.use_llm.get()

        def work():
            try:
                with mss.mss() as sct:
                    screenshot = sct.grab(sct.monitors[1])
                img = np.array(screenshot)
                img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
                detected = self._pipeline_ocr_and_optional_llm(img, use_llm, is_full_screen=True)
                self.root.after(0, lambda d=detected: self._finish_capture(d, None))
            except Exception as e:
                self.root.after(0, lambda err=e: self._finish_capture(None, err))

        threading.Thread(target=work, daemon=True).start()

    def capture_active_window(self):
        if self._ocr_busy:
            return
        self._set_busy(True)
        self.status_label.config(text="Capturando janela ativa (OCR em segundo plano)...")
        use_llm = self.use_llm.get()

        def work():
            try:
                hwnd = win32gui.GetForegroundWindow()
                rect = win32gui.GetWindowRect(hwnd)
                x, y, right, bottom = rect
                monitor = {"top": y, "left": x, "width": right - x, "height": bottom - y}
                with mss.mss() as sct:
                    screenshot = sct.grab(monitor)
                img = np.array(screenshot)
                img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
                detected = self._pipeline_ocr_and_optional_llm(img, use_llm, is_full_screen=False)
                self.root.after(0, lambda d=detected: self._finish_capture(d, None))
            except Exception as e:
                self.root.after(0, lambda err=e: self._finish_capture(None, err))

        threading.Thread(target=work, daemon=True).start()

    def _finish_capture(self, results, error):
        self._set_busy(False)
        if error is not None:
            self.status_label.config(text=f"Erro na captura/OCR: {error}")
            self.text_area.delete(1.0, tk.END)
            self.text_area.insert(1.0, str(error))
            return
        self._last_results = results
        self._web_store.publish(results)
        self.display_results(results)

    def analyze_image(self, img, is_full_screen: bool = False):
        img = _crop_image_bgr(img, is_full_screen=is_full_screen)
        filtered, raw_text = _ocr_best_of_variants(img, _TESSERACT_LANG, _TESSERACT_CONFIG)
        questions, options_by_question = _parse_questions_structured(filtered)
        summaries = _summarize_items(questions, options_by_question)


        filtered_text = "\n".join(ln["text"] for ln in (filtered or []))
        llm_compact = _build_llm_compact_block(questions, options_by_question)
        return {
            "full_text": raw_text,
            "filtered_text": filtered_text,
            "llm_compact_text": llm_compact,
            "questions": questions,
            "options_by_question": options_by_question,
            "summaries": summaries,
            "total_questions": len(questions),
        }

    def _pipeline_ocr_and_optional_llm(self, img, use_llm: bool, is_full_screen: bool = False):
        t0 = time.perf_counter()
        detected = self.analyze_image(img, is_full_screen=is_full_screen)
        detected["timings_ms"] = {"ocr_parse": int((time.perf_counter() - t0) * 1000)}
        want = use_llm
        detected["ollama_used"] = want
        detected["ollama_parsed"] = None
        detected["ollama_error"] = None
        detected["ollama_raw"] = None
        detected["ollama_skipped_offline_full"] = False
        if not want:
            return detected

        qs = detected.get("questions") or []
        obq = detected.get("options_by_question") or []
        if _env_flag("SKIP_LLM_WHEN_OFFLINE_FULL") and _all_offline_covers_every_question(qs, obq):
            synthetic_items: list[dict] = []
            for i, q in enumerate(qs):
                opts = list(obq[i]) if i < len(obq) else []
                letter, _txt, why = _offline_fallback_suggestion(q, opts)
                synthetic_items.append(
                    {
                        "question": q,
                        "options": opts,
                        "suggested": letter or "",
                        "confidence": "alta",
                        "note": (why or "").strip() or "Heurística offline (sem chamada ao modelo).",
                    }
                )
            parsed_syn: dict = {"items": synthetic_items}
            enriched = enrich_parsed_items(parsed_syn)
            detected["ollama_parsed"] = enriched or parsed_syn
            detected["ollama_skipped_offline_full"] = True
            return detected

        self.root.after(0, lambda: self.status_label.config(text="AirLLM: refinando texto (JSON)..."))
        try:
            t_llm0 = time.perf_counter()
            llm_input = (detected.get("llm_compact_text") or "").strip()
            if len(llm_input) < 30:
                llm_input = detected.get("filtered_text") or detected["full_text"]
            detected["ollama_used"] = False
            detected["ollama_backend"] = "airllm"
            self.root.after(0, lambda: self.status_label.config(text="AirLLM: refinando texto (JSON)..."))
            provider = self._get_llm_provider()

            two_pass = _env_flag_default("LLM_TWO_PASS", True)
            if two_pass:
                # Passo 1: extrair estrutura
                p1 = build_quiz_extract_prompt(llm_input)
                r1 = provider.complete(p1)
                parsed1, err1 = parse_items_from_llm_response(r1.raw)
                if not parsed1 or err1:
                    detected["ollama_raw"] = r1.raw
                    detected["ollama_error"] = err1 or "Falha no passo 1 (extract)."
                    return detected

                # Passo 2: sugerir alternativa em cima do JSON limpo
                p2 = build_quiz_suggest_prompt(json.dumps(parsed1, ensure_ascii=False))
                r2 = provider.complete(p2)
                detected["ollama_raw"] = r2.raw
                parsed2, err2 = parse_items_from_llm_response(r2.raw)
                if not parsed2 or err2:
                    detected["ollama_error"] = err2 or "Falha no passo 2 (suggest)."
                    detected["ollama_parsed"] = parsed1
                    return detected

                # Merge: injeta suggested/confidence/note no parsed1
                items1 = parsed1.get("items") if isinstance(parsed1.get("items"), list) else []
                items2 = parsed2.get("items") if isinstance(parsed2.get("items"), list) else []
                if isinstance(items1, list) and isinstance(items2, list):
                    for i in range(min(len(items1), len(items2))):
                        if isinstance(items1[i], dict) and isinstance(items2[i], dict):
                            for k in ("suggested", "confidence", "note"):
                                if k in items2[i] and items2[i].get(k) is not None:
                                    items1[i][k] = items2[i].get(k)
                parsed, err = parsed1, None
            else:
                prompt = build_quiz_refine_prompt(llm_input)
                res = provider.complete(prompt)
                detected["ollama_raw"] = res.raw
                parsed, err = parse_items_from_llm_response(res.raw)

            detected["timings_ms"]["llm"] = int((time.perf_counter() - t_llm0) * 1000)
            if parsed and not err:
                parsed = enrich_parsed_items(parsed)
                parsed = align_llm_to_ocr(parsed, detected.get("options_by_question") or [])
            detected["ollama_parsed"] = parsed
            detected["ollama_error"] = err
        except LLMUnavailableError as e:
            detected["ollama_error"] = str(e)
        except (ConnectionError, TimeoutError, ValueError) as e:
            detected["ollama_error"] = str(e)
        except Exception as e:
            detected["ollama_error"] = f"Erro inesperado no LLM: {e}"
        return detected

    def _configure_result_tags(self):
        ta = self.text_area
        ta.tag_configure("title", font=("Segoe UI", 11, "bold"), foreground="#38bdf8")
        ta.tag_configure("rule", foreground="#475569")
        ta.tag_configure("qhead", font=("Consolas", 10, "bold"), foreground="#94a3b8")
        ta.tag_configure("question", font=("Consolas", 10), foreground="#f8fafc")
        ta.tag_configure("olabel", font=("Consolas", 10, "bold"), foreground="#a5b4fc")
        ta.tag_configure("option", font=("Consolas", 10), foreground="#e2e8f0")
        ta.tag_configure("ans_title", font=("Segoe UI", 11, "bold"), foreground="#4ade80")
        ta.tag_configure("ans_key", font=("Consolas", 11, "bold"), foreground="#fbbf24")
        ta.tag_configure("ans_body", font=("Consolas", 10), foreground="#fef08a")
        ta.tag_configure("meta", font=("Consolas", 9), foreground="#64748b")
        ta.tag_configure("warn", font=("Consolas", 9), foreground="#f87171")
        ta.tag_configure("tech", font=("Consolas", 8), foreground="#52525b")

    @staticmethod
    def _option_text_for_suggestion(suggested: str, options: list) -> tuple[str, str]:
        """Retorna (rótulo, texto completo da opção) a partir da sugestão da IA."""
        sug = (suggested or "").strip()
        if not sug or not options:
            return "", ""
        opts = [str(o).strip() for o in options if str(o).strip()]
        if not opts:
            return sug, ""
        s_up = sug.upper()
        for o in opts:
            ou = o.upper()
            if ou.startswith(s_up + ".") or ou.startswith(s_up + ")"):
                return sug, o
            if len(sug) == 1 and sug.isalpha() and ou.startswith(sug.upper() + "."):
                return sug, o
        if sug.isdigit():
            i = int(sug) - 1
            if 0 <= i < len(opts):
                return sug, opts[i]
        if len(sug) == 1 and "A" <= s_up <= "Z":
            i = ord(s_up) - ord("A")
            if 0 <= i < len(opts):
                return sug, opts[i]
        for i, o in enumerate(opts):
            if o.lstrip().startswith(sug):
                return sug, o
        return sug, ""

    @staticmethod
    def _final_answer_for_question(results: dict, idx: int) -> tuple[str | None, str | None, str]:
        """(rótulo sugerido, texto da opção, confiança ou '')."""
        qs = results.get("questions") or []
        opts_by_q = results.get("options_by_question") or []
        if idx < 0 or idx >= len(qs):
            return None, None, ""
        q = qs[idx]
        opts = opts_by_q[idx] if idx < len(opts_by_q) else []

        parsed = results.get("ollama_parsed")
        items = (
            parsed.get("items")
            if parsed and isinstance(parsed.get("items"), list)
            else []
        )
        if items and idx < len(items) and isinstance(items[idx], dict):
            it = items[idx]
            sug = (it.get("suggested") or "").strip()
            opts_llm = it.get("options") if isinstance(it.get("options"), list) else []
            label, full_opt = InvisibleScreenCapture._option_text_for_suggestion(sug, opts_llm)
            conf = (it.get("confidence") or "").strip() if isinstance(it, dict) else ""
            if label and full_opt:
                return label, full_opt, conf
            if label:
                return label, full_opt or "", conf

        letter, text, _ = _offline_fallback_suggestion(q, opts)
        if letter and text:
            return letter, text, ""

        kw = _infer_suggested_by_keywords(q, opts)
        if kw and opts:
            _lab, full = InvisibleScreenCapture._option_text_for_suggestion(kw, opts)
            if full:
                return kw, full, "baixa"
            if kw:
                return kw, "", "baixa"

        return None, None, ""

    def _display_results_compact(self, results):
        ta = self.text_area

        qs = results.get("questions") or []
        show_conf = _env_flag("UI_SHOW_CONFIDENCE")

        ta.insert(tk.END, "\n")
        if not qs:
            ta.insert(tk.END, "\n  (sem perguntas detectadas)\n", ("meta",))
            ta.see(1.0)
            return

        for idx in range(min(len(qs), 10)):
            q = qs[idx]
            label, full_opt, conf = self._final_answer_for_question(results, idx)
            ta.insert(tk.END, f"\n  Pergunta {idx + 1}\n", ("qhead",))
            ta.insert(tk.END, f"  {q}\n\n", ("question",))
            ta.insert(tk.END, "  Resposta: ", ("ans_title",))
            if label and full_opt:
                ta.insert(tk.END, f"{label}) {full_opt}\n", ("ans_body",))
            elif label:
                ta.insert(tk.END, f"{label}) (texto não detectado)\n", ("ans_key",))
            else:
                ta.insert(
                    tk.END,
                    "(sem sugestão)\n",
                    ("meta",),
                )
            if show_conf and conf:
                ta.insert(tk.END, f"  Confiança: {conf}\n", ("meta",))

        ta.insert(tk.END, "\n", ())
        ta.see(1.0)

    def display_results(self, results):
        self.text_area.delete(1.0, tk.END)
        self._configure_result_tags()
        if self.compact_quiz.get():
            self._display_results_compact(results)
            n_llm = 0
            if results.get("ollama_parsed") and isinstance(
                results["ollama_parsed"].get("items"), list
            ):
                n_llm = len(results["ollama_parsed"]["items"])
            base = f"✅ {results['total_questions']} pergunta(s)"
            if results.get("ollama_skipped_offline_full"):
                self.status_label.config(text=f"{base} | Respostas offline (LLM não chamado)")
            elif results.get("ollama_used"):
                self.status_label.config(text=f"{base} | IA: {n_llm}")
            else:
                self.status_label.config(text=base)
            return

        ta = self.text_area

        ta.insert(tk.END, "\n")
        ta.insert(tk.END, "  Quiz na tela (OCR)\n", ("title",))
        ta.insert(tk.END, "  " + "─" * 40 + "\n", ("rule",))

        qs = results.get("questions") or []
        opts_by_q = results.get("options_by_question") or []

        if qs:
            for idx, q in enumerate(qs[:10], start=1):
                ta.insert(tk.END, f"\n  Pergunta {idx}\n", ("qhead",))
                ta.insert(tk.END, f"  {q}\n\n", ("question",))
                opts = opts_by_q[idx - 1] if idx <= len(opts_by_q) else []
                if opts:
                    for oi, opt in enumerate(opts[:10], start=1):
                        letter = chr(64 + oi)
                        ta.insert(tk.END, f"    {letter})  ", ("olabel",))
                        ta.insert(tk.END, f"{opt}\n", ("option",))
                else:
                    ta.insert(tk.END, "    (alternativas não detectadas)\n", ("meta",))
        else:
            ta.insert(tk.END, "\n  Nenhuma pergunta detectada no OCR.\n", ("warn",))

        ta.insert(tk.END, "\n")
        ta.insert(tk.END, "  " + "═" * 40 + "\n", ("rule",))
        ta.insert(tk.END, "\n  Resposta sugerida (IA local)\n", ("ans_title",))
        ta.insert(tk.END, "  Sugestão automática — valide com o material da prova.\n\n", ("meta",))
        ta.insert(tk.END, "  " + "─" * 40 + "\n\n", ("rule",))

        ia_showed_suggestion = False
        parsed = results.get("ollama_parsed")
        if parsed and isinstance(parsed.get("items"), list) and parsed["items"]:
            for it in parsed["items"][:10]:
                sug = (it.get("suggested") or "").strip()
                conf = (it.get("confidence") or "").strip()
                note = (it.get("note") or "").strip()
                opts_llm = it.get("options") if isinstance(it.get("options"), list) else []
                label, full_opt = self._option_text_for_suggestion(sug, opts_llm)
                if sug:
                    ta.insert(tk.END, "  ► Alternativa: ", ("ans_key",))
                    ta.insert(tk.END, f"{sug}\n", ("ans_key",))
                    ia_showed_suggestion = True
                else:
                    ta.insert(tk.END, "  ► A IA não indicou alternativa (incerto).\n", ("meta",))
                if full_opt:
                    ta.insert(tk.END, "  ", ("meta",))
                    ta.insert(tk.END, f"{full_opt}\n", ("ans_body",))
                elif sug:
                    ta.insert(tk.END, "  (texto da opção não casou com a lista da IA)\n", ("meta",))
                if conf:
                    ta.insert(tk.END, f"  Confiança: {conf}\n", ("meta",))
                if note:
                    ta.insert(tk.END, f"  Nota: {note}\n", ("meta",))
                ta.insert(tk.END, "\n")
        elif results.get("ollama_used"):
            err = results.get("ollama_error") or "Falha desconhecida."
            ta.insert(tk.END, f"  {err}\n", ("warn",))
            raw = results.get("ollama_raw")
            if raw:
                snip = raw[:900] + ("..." if len(raw) > 900 else "")
                ta.insert(tk.END, "\n  Trecho bruto do modelo:\n", ("meta",))
                ta.insert(tk.END, f"  {snip}\n", ("tech",))
        else:
            ta.insert(tk.END, "  IA desligada — usando pista offline se reconhecer a pergunta.\n", ("meta",))

        if not ia_showed_suggestion and qs:
            ta.insert(tk.END, "\n")
            ta.insert(tk.END, "  ► Resposta (sem IA — reconhecimento local)\n", ("ans_title",))
            ta.insert(tk.END, "  Heurística no código; só cobre alguns enunciados típicos.\n\n", ("meta",))
            any_offline = False
            for idx, q in enumerate(qs[:6], start=1):
                opts = opts_by_q[idx - 1] if idx <= len(opts_by_q) else []
                letter, text, why = _offline_fallback_suggestion(q, opts)
                if letter and text:
                    any_offline = True
                    ta.insert(tk.END, f"  Pergunta {idx} → alternativa ", ("ans_key",))
                    ta.insert(tk.END, f"{letter}\n", ("ans_key",))
                    ta.insert(tk.END, f"  {text}\n", ("ans_body",))
                    ta.insert(tk.END, f"  {why}\n\n", ("meta",))
            if not any_offline:
                ta.insert(tk.END, "  Nenhum padrão local bateu nesta captura. Use a IA (AirLLM) para refinar.\n", ("meta",))

        ta.insert(tk.END, "\n")
        ta.insert(tk.END, "  " + "─" * 40 + "\n", ("rule",))
        ta.insert(tk.END, "  Referência técnica (OCR filtrado)\n", ("qhead",))
        ta.insert(tk.END, "  " + "─" * 40 + "\n", ("rule",))

        ft = results.get("filtered_text") or ""
        if ft:
            max_show = int(os.environ.get("OCR_DISPLAY_FILTERED_MAX", "12000"))
            if len(ft) <= max_show:
                ta.insert(tk.END, f"  ({len(ft)} caracteres, completo)\n\n", ("tech",))
                ta.insert(tk.END, ft + "\n", ("tech",))
            else:
                ta.insert(tk.END, f"  ({len(ft)} caracteres, primeiros {max_show})\n\n", ("tech",))
                ta.insert(tk.END, ft[:max_show] + "\n... (truncado)\n", ("tech",))

        ta.insert(tk.END, "\n  OCR bruto (amostra)\n", ("qhead",))
        raw_full = results.get("full_text") or ""
        ocr_sample = 280
        ta.insert(tk.END, (raw_full[:ocr_sample] + ("..." if len(raw_full) > ocr_sample else "") + "\n"), ("tech",))

        ta.see(1.0)

        n_llm = 0
        if results.get("ollama_parsed") and isinstance(results["ollama_parsed"].get("items"), list):
            n_llm = len(results["ollama_parsed"]["items"])
        base = f"✅ {results['total_questions']} pergunta(s)"
        if results.get("ollama_skipped_offline_full"):
            self.status_label.config(text=f"{base} | Respostas offline (LLM não chamado)")
        elif results.get("ollama_used"):
            self.status_label.config(text=f"{base} | IA: {n_llm}")
        else:
            self.status_label.config(text=base)

    def clear_text(self):
        self.text_area.delete(1.0, tk.END)
        self.status_label.config(text="Pronto para captura")

    def copy_results(self):
        """Copia o resumo atual (modo compacto) ou o texto exibido."""
        try:
            txt = self.text_area.get("1.0", tk.END).strip()
            if not txt:
                return
            self.root.clipboard_clear()
            self.root.clipboard_append(txt)
            self.status_label.config(text="Copiado para a área de transferência.")
        except Exception as e:
            self.status_label.config(text=f"Falha ao copiar: {e}")


if __name__ == "__main__":
    print("Programa iniciado!")
    print("A janela deve aparecer para você e ficar oculta na maioria dos compartilhamentos (Windows 10/11 recente).")
    print("Hotkeys: Ctrl+Shift+C (esconder/mostrar na sua tela), S (captura tela), W (janela ativa)")
    print("Requer Windows 10 versão 2004 ou superior para exclusão no compartilhamento.")
    print("AirLLM: marque o checkbox na UI e configure AIRLLM_MODEL (repo HF ou caminho local).")
    print("OCR: TESSERACT_LANG=padrão eng+por; recorte tela inteira: OCR_CROP_LEFT/BOTTOM/RIGHT_FRAC (janela: OCR_CROP_WINDOW_*).")
    print("AirLLM: AIRLLM_DEVICE=cpu (padrão); AIRLLM_MAX_INPUT_TOKENS e AIRLLM_MAX_NEW_TOKENS ajustam custo/latência.")
    print("Velocidade/UI: OCR_FAST=1 (OCR mais leve); UI_COMPACT_QUIZ=1 (lista compacta ao abrir);")
    print("  LLM_TWO_PASS=1 (padrão) melhora a estrutura quando OCR está ruidoso;")
    print("  UI_SHOW_CONFIDENCE=1 (no modo compacto, mostra linha de confiança quando existir).")
    print(
        "Exclusão de captura: WDA_EXCLUDEFROMCAPTURE em HWND do Tk + GetAncestor(ROOT); "
        "EXCLUDE_TRY_LAYERED=1 força WS_EX_LAYERED no candidato e tenta de novo (opcional)."
    )
    print(
        "Atenção: não há garantia de ocultar em todo app/modo (Discord/Teams/OBS podem ignorar ou preto/vazio variar)."
    )
    print(
        "Web local: WEB_ENABLE=1 inicia servidor acessível na rede (celular). "
        "WEB_PORT=8765 padrão, WEB_HISTORY=25. WEB_LOCAL_ONLY=1 restringe ao PC. "
        "Botão 'Abrir Web' na UI também inicia sob demanda."
    )

    if _env_flag("SHOW_EDU_SECURITY_BLURB"):
        print("")
        print("--- Aula IA / segurança (instrutores) ---")
        print(
            "SetWindowDisplayAffinity (WDA_EXCLUDEFROMCAPTURE) pode fazer esta janela sumir em "
            "alguns fluxos de captura (ex.: compartilhamento em Teams, Zoom ou Discord). "
            "Não é garantido: depende de versão do app, modo tela inteira vs. janela, drivers e SO."
        )
        print(
            "Use em sala para conscientização (privacidade, limites da captura). "
            "Demonstração técnica ≠ justificar fraude em provas ou monitoramento proibido."
        )
        print("Roteiro sugerido: testar cada app, anotar diferenças, discutir implicações em ambientes corporativos.")
        print("---")

    app = InvisibleScreenCapture()
