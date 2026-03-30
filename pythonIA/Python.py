import ctypes
import os
import re
import threading
import tkinter as tk
from tkinter import ttk

import cv2
import keyboard
import mss
import numpy as np
import pytesseract
import win32gui

from ollama_client import (
    build_quiz_refine_prompt,
    ollama_complete,
    parse_items_from_llm_response,
)

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

_TESSERACT_CONFIG = os.environ.get("TESSERACT_CONFIG", r"--oem 3 --psm 6")
_TESSERACT_LANG = os.environ.get("TESSERACT_LANG", "eng+por")

_OPTION_LINE_RE = re.compile(
    r"^(\d{1,2}|[A-Ea-e])[\.\)\:\-\s]\s*\S",
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
_UNNUM_MCQ_START_RE = re.compile(r"^(An API\b|A proxy\b|A service\b)", re.I)
_DATE_TASKBAR_RE = re.compile(r"\b\d{1,2}/\d{1,2}/\d{2,4}\b")
_LMS_MENU_ONLY_RE = re.compile(
    r"^(KEYBOARD\s+NAVIGATION|Dashboard|Calendar|Inbox|Account|Modules?|Grades?|"
    r"Discussions?|Announcements?|Courses?|Home|Help)\s*$",
    re.I,
)
_OPTION_BLOCK_SPLIT_RE = re.compile(
    r"(?m)^\s*[oO,]*\s*([1-6])\.\s+(.+?)(?=^\s*[oO,]*\s*[1-6]\.\s+|\Z)",
    re.DOTALL,
)
_MAX_OPTIONS_PER_QUESTION = 6


def _env_flag(name: str) -> bool:
    return os.environ.get(name, "").strip().lower() in ("1", "true", "yes", "on")


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
    if re.search(r"\b[1-4]\.\s+[A-Za-z]", glued):
        parts = re.split(r"(?=\b[1-4]\.\s+)", glued)
        out: list[str] = []
        for p in parts:
            p = p.strip()
            if re.match(r"^[1-4]\.\s+\S", p) and len(p) > 15 and not _is_junk_continuation_line(p):
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
        if re.match(r"^An \w+", c, re.I) and len(c) > 22:
            opts.append(c)
            continue
        if opts and _is_continuation_mcq_line(c):
            opts[-1] = (opts[-1] + " " + c).strip()
            continue
        if not opts:
            continue
        if len(c) > 35 and c[0].isupper() and re.search(
            r"\b(API|proxy|server|client|HTTP|REST)\b", c, re.I
        ):
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
            and len(c) <= 70
            and not _line_looks_like_option_start(c)
            and not re.match(
                r"^(Which|What|Select|Choose|How|When|Where|Why|Qual|Como|Que)\b",
                c,
                re.I,
            )
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


def _hwnd_exclude_from_screen_capture(hwnd: int) -> bool:
    if os.name != "nt" or hwnd <= 0:
        return False
    try:
        user32 = ctypes.windll.user32
        user32.SetWindowDisplayAffinity.argtypes = [ctypes.c_void_p, ctypes.c_uint]
        user32.SetWindowDisplayAffinity.restype = ctypes.c_int
        return bool(user32.SetWindowDisplayAffinity(hwnd, _WDA_EXCLUDEFROMCAPTURE))
    except Exception:
        return False


def _preprocess_for_ocr(img_bgr: np.ndarray) -> np.ndarray:
    """Escala leve + contraste: melhora leitura de texto em páginas web."""
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape[:2]
    fast = _env_flag("OCR_FAST")
    cap = 1.35 if fast else 2.0
    if w < 1400:
        scale = min(cap, 1400 / max(w, 1))
        gray = cv2.resize(gray, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    gray = clahe.apply(gray)
    if fast:
        return gray
    return cv2.bilateralFilter(gray, d=5, sigmaColor=45, sigmaSpace=45)


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
    Quando não há Ollama: heurísticas mínimas por texto da pergunma + alternativas (sem rede).
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

        self.root.attributes("-alpha", 0.95)
        self.root.attributes("-topmost", True)
        self.root.overrideredirect(True)

        # mss no Windows usa estado por thread — não reutilizar instância fora da thread que captura
        self._ocr_busy = False

        keyboard.add_hotkey("ctrl+shift+c", lambda: self.root.after(0, self.toggle_visibility))
        keyboard.add_hotkey("ctrl+shift+s", lambda: self.root.after(0, self.capture_screen))
        keyboard.add_hotkey("ctrl+shift+w", lambda: self.root.after(0, self.capture_active_window))

        self.root.protocol("WM_DELETE_WINDOW", self.on_closing)

        self.setup_ui()
        self.root.after(0, self._apply_exclude_from_capture)
        self.root.mainloop()

    def _apply_exclude_from_capture(self):
        self.root.update_idletasks()
        hwnd = int(self.root.winfo_id())
        ok = _hwnd_exclude_from_screen_capture(hwnd)
        if not ok and os.name == "nt":
            self.status_label.config(
                text="Aviso: exclusão de captura não aplicada (Windows 10 2004+ necessário) | Hotkeys: Ctrl+Shift+C, S, W"
            )

    def setup_ui(self):
        main_frame = ttk.Frame(self.root)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        self.status_label = ttk.Label(
            main_frame,
            text="Visível para você | Oculto no compartilhamento de tela | Ctrl+Shift+C esconder/mostrar aqui",
            font=("Arial", 9),
        )
        self.status_label.pack(pady=5)

        self.use_ollama = tk.BooleanVar(value=False)
        self.chk_ollama = ttk.Checkbutton(
            main_frame,
            text="Refinar com Ollama após captura (gemma3:4b)",
            variable=self.use_ollama,
        )
        self.chk_ollama.pack(anchor=tk.W, pady=(0, 2))

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
        self.btn_clear = ttk.Button(btn_frame, text="Limpar", command=self.clear_text)
        self.btn_clear.pack(side=tk.LEFT, padx=2)

    def _set_busy(self, busy: bool):
        self._ocr_busy = busy
        state = "disabled" if busy else "normal"
        self.btn_full.configure(state=state)
        self.btn_win.configure(state=state)
        self.chk_ollama.configure(state=state)
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
        use_llm = self.use_ollama.get()

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
        use_llm = self.use_ollama.get()

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
        self.display_results(results)

    def analyze_image(self, img, is_full_screen: bool = False):
        img = _crop_image_bgr(img, is_full_screen=is_full_screen)
        proc = _preprocess_for_ocr(img)
        try:
            text = pytesseract.image_to_string(proc, lang=_TESSERACT_LANG, config=_TESSERACT_CONFIG)
        except Exception:
            text = pytesseract.image_to_string(proc, lang="eng", config=_TESSERACT_CONFIG)

        raw_lines = [line.strip() for line in text.split("\n")]
        lines_all = [line for line in raw_lines if line]
        lines = _filter_ocr_lines(lines_all)

        # Palavras que iniciam pergunta sem gerar falso positivo em continuações ("when they submit...")
        _safe_q_patterns = [
            "Qual",
            "Como",
            "Que",
            "Q:",
            "Pergunta:",
            "Which",
            "What",
            "Select",
            "Choose",
        ]
        _q_start_re = re.compile(
            r"^(\d+\.\s*)?(Qual|Como|Que|Q:|Pergunta:|Which|What|Select|Choose)\b",
            re.I,
        )
        # How/When/Where/Why só no início da linha + auxiliar (evita "When they submit...")
        _wh_aux_re = re.compile(
            r"^(\d+\.\s*)?(How|When|Where|Why)\s+"
            r"(do|does|did|is|are|can|could|should|would|will|must|many|much|to)\b",
            re.I,
        )

        def _is_question_line(line: str) -> bool:
            s = line.strip()
            if _looks_like_url_line(s):
                return False
            if _BREADCRUMB_ASSIGNMENTS_RE.search(s) and "Which" not in s and "What" not in s:
                return False
            # Continuação de enunciado quebrado pelo OCR (ex.: "when they submit an order.")
            if s and s[0].islower() and "?" not in s:
                if not re.match(
                    r"^(which|what|how|when|where|why|select|choose|qual|como|quando|onde)\b",
                    s,
                    re.I,
                ):
                    return False
            # Heurística: só trata como pergunta quando o prefixo aparece no início
            # (evita falso positivo quando palavras-chave aparecem “no meio” da alternativa).
            if _q_start_re.match(s):
                return True
            if _wh_aux_re.match(s):
                return True
            if "?" in s and not _URL_FRAGMENT_RE.search(s):
                if len(s) < 260 and re.search(
                    r"\b(describe|best|correct|true|false|select|choose|suggests|option|phrase)\b",
                    s,
                    re.I,
                ):
                    return True
            return False

        questions: list[str] = []
        options_by_question: list[list[str]] = []

        i = 0
        while i < len(lines):
            line = lines[i]
            if _is_question_line(line):
                q_clean = _clean_question_line(line)
                questions.append(q_clean if q_clean else line)

                block_lines: list[str] = []
                j = i + 1
                while j < len(lines):
                    candidate = lines[j]
                    if _is_question_line(candidate):
                        break
                    block_lines.append(candidate)
                    j += 1

                block_text = "\n".join(block_lines)
                split_opts = _split_options_from_block(block_text, _MAX_OPTIONS_PER_QUESTION)
                fb_opts = _fallback_collect_options(block_lines, _MAX_OPTIONS_PER_QUESTION)
                un_opts = _split_unnumbered_mcq_lines(block_lines, _MAX_OPTIONS_PER_QUESTION)
                opts = _pick_best_options(split_opts, fb_opts, un_opts)

                options_by_question.append(opts)
                i = j
                continue
            i += 1

        filtered_text = "\n".join(lines)
        llm_compact = _build_llm_compact_block(questions, options_by_question)
        return {
            "full_text": text,
            "filtered_text": filtered_text,
            "llm_compact_text": llm_compact,
            "questions": questions,
            "options_by_question": options_by_question,
            "total_questions": len(questions),
        }

    def _pipeline_ocr_and_optional_llm(self, img, use_llm: bool, is_full_screen: bool = False):
        detected = self.analyze_image(img, is_full_screen=is_full_screen)
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
        if _env_flag("SKIP_OLLAMA_WHEN_OFFLINE_FULL") and _all_offline_covers_every_question(qs, obq):
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

        self.root.after(0, lambda: self.status_label.config(text="Ollama: refinando texto (JSON)..."))
        try:
            llm_input = (detected.get("llm_compact_text") or "").strip()
            if len(llm_input) < 30:
                llm_input = detected.get("filtered_text") or detected["full_text"]
            prompt = build_quiz_refine_prompt(llm_input)
            raw = ollama_complete(prompt)
            detected["ollama_raw"] = raw
            parsed, err = parse_items_from_llm_response(raw)
            if parsed and not err:
                parsed = enrich_parsed_items(parsed)
            detected["ollama_parsed"] = parsed
            detected["ollama_error"] = err
        except (ConnectionError, TimeoutError, ValueError) as e:
            detected["ollama_error"] = str(e)
        except Exception as e:
            detected["ollama_error"] = f"Erro inesperado no Ollama: {e}"
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
                self.status_label.config(text=f"{base} | Respostas offline (Ollama não chamado)")
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
            ta.insert(tk.END, "  Ollama desligado — usando pista offline se reconhecer a pergunma.\n", ("meta",))

        if not ia_showed_suggestion and qs:
            ta.insert(tk.END, "\n")
            ta.insert(tk.END, "  ► Resposta (sem Ollama — reconhecimento local)\n", ("ans_title",))
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
                ta.insert(tk.END, "  Nenhum padrão local bateu nesta captura. Use Ollama com modelo menor\n", ("meta",))
                ta.insert(tk.END, "  (ex.: gemma3:4b) ou marque o checkbox quando a RAM permitir.\n", ("meta",))

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
            self.status_label.config(text=f"{base} | Respostas offline (Ollama não chamado)")
        elif results.get("ollama_used"):
            self.status_label.config(text=f"{base} | IA: {n_llm}")
        else:
            self.status_label.config(text=base)

    def clear_text(self):
        self.text_area.delete(1.0, tk.END)
        self.status_label.config(text="Status: Pronto para captura")


if __name__ == "__main__":
    print("🚀 Programa iniciado!")
    print("A janela deve aparecer para você e ficar oculta na maioria dos compartilhamentos (Windows 10/11 recente).")
    print("Hotkeys: Ctrl+Shift+C (esconder/mostrar na sua tela), S (captura tela), W (janela ativa)")
    print("Requer Windows 10 versão 2004 ou superior para exclusão no compartilhamento.")
    print("Ollama opcional: marque o checkbox na UI; env OLLAMA_HOST / OLLAMA_MODEL (padrão gemma3:4b).")
    print("OCR: TESSERACT_LANG=padrão eng+por; recorte tela inteira: OCR_CROP_LEFT/BOTTOM/RIGHT_FRAC (janela: OCR_CROP_WINDOW_*).")
    print("Ollama: OLLAMA_NUM_CTX (padrão 2048, 0=omitir); OLLAMA_NUM_PREDICT (padrão 768); RAM baixa use modelo menor.")
    print("Velocidade/UI: OCR_FAST=1 (OCR mais leve); UI_COMPACT_QUIZ=1 (lista compacta ao abrir);")
    print("  SKIP_OLLAMA_WHEN_OFFLINE_FULL=1 (pula API se heurística offline cobrir todas as perguntas);")
    print("  UI_SHOW_CONFIDENCE=1 (no modo compacto, mostra linha de confiança quando existir).")

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
