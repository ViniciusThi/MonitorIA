"""Backend opcional: AirLLM (HuggingFace) para inferência mais eficiente em memória.

Referência: https://github.com/lyogavin/airllm
"""

from __future__ import annotations

import os
from dataclasses import dataclass

from llm_providers import LLMResult, LLMUnavailableError


DEFAULT_AIRLLM_MODEL = os.environ.get("AIRLLM_MODEL", "").strip()
DEFAULT_AIRLLM_MAX_NEW_TOKENS = int(os.environ.get("AIRLLM_MAX_NEW_TOKENS", "512"))
DEFAULT_AIRLLM_COMPRESSION = os.environ.get("AIRLLM_COMPRESSION", "").strip()  # 4bit|8bit|""(none)
DEFAULT_AIRLLM_DEVICE = os.environ.get("AIRLLM_DEVICE", "cpu").strip()  # cpu|cuda|mps (se suportado)
DEFAULT_AIRLLM_MAX_INPUT_TOKENS = int(os.environ.get("AIRLLM_MAX_INPUT_TOKENS", "3072"))
DEFAULT_AIRLLM_HF_TOKEN = os.environ.get("HF_TOKEN", os.environ.get("HUGGINGFACE_TOKEN", "")).strip()
DEFAULT_AIRLLM_SHARDS_PATH = os.environ.get("AIRLLM_SHARDS_PATH", "").strip()
DEFAULT_AIRLLM_PROFILING = os.environ.get("AIRLLM_PROFILING", "").strip().lower() in ("1", "true", "yes", "on")


def _require_airllm():
    try:
        from airllm import AutoModel  # type: ignore
        return AutoModel
    except Exception as e:  # pragma: no cover
        raise LLMUnavailableError(
            "AirLLM não está disponível neste ambiente. "
            "Instale com `pip install airllm` e configure AIRLLM_MODEL "
            "(ex.: `garage-bAInd/Platypus2-70B-instruct` ou um modelo local). "
            f"Detalhe: {e}"
        ) from e


@dataclass
class AirLLMConfig:
    model_id_or_path: str
    max_new_tokens: int = DEFAULT_AIRLLM_MAX_NEW_TOKENS
    compression: str = DEFAULT_AIRLLM_COMPRESSION
    device: str = DEFAULT_AIRLLM_DEVICE
    max_input_tokens: int = DEFAULT_AIRLLM_MAX_INPUT_TOKENS
    hf_token: str = DEFAULT_AIRLLM_HF_TOKEN
    layer_shards_saving_path: str = DEFAULT_AIRLLM_SHARDS_PATH
    profiling_mode: bool = DEFAULT_AIRLLM_PROFILING


class AirLLMProvider:
    name = "airllm"

    def __init__(self, cfg: AirLLMConfig):
        if not cfg.model_id_or_path:
            raise LLMUnavailableError(
                "AIRLLM_MODEL não configurado. Defina AIRLLM_MODEL com o repo ID do HuggingFace "
                "ou caminho local do modelo."
            )
        self._cfg = cfg
        self._model = None

    def _get_model(self):
        if self._model is not None:
            return self._model
        AutoModel = _require_airllm()
        kwargs: dict = {}

        comp = (self._cfg.compression or "").strip()
        if comp:
            kwargs["compression"] = comp

        if self._cfg.profiling_mode:
            kwargs["profiling_mode"] = True

        if (self._cfg.layer_shards_saving_path or "").strip():
            kwargs["layer_shards_saving_path"] = self._cfg.layer_shards_saving_path

        if (self._cfg.hf_token or "").strip():
            kwargs["hf_token"] = self._cfg.hf_token

        # Nota: AirLLM faz shard/layer-splitting e cache em disco automaticamente.
        self._model = AutoModel.from_pretrained(self._cfg.model_id_or_path, **kwargs)
        return self._model

    def warmup(self) -> None:
        """Pré-carrega modelo (útil para reduzir latência da 1ª chamada)."""
        _ = self._get_model()

    def complete(self, prompt: str) -> LLMResult:
        model = self._get_model()

        # Tokenização + geração simples (modo "completion").
        # Para chat, o prompt já deve vir formatado (o seu build_quiz_refine_prompt já é bem direto).
        # CPU/32GB: permitir mais contexto por padrão; truncar para estabilidade
        max_in = max(256, int(self._cfg.max_input_tokens))
        toks = model.tokenizer(
            [prompt],
            return_tensors="pt",
            return_attention_mask=False,
            truncation=True,
            max_length=max_in,
            padding=False,
        )

        input_ids = toks["input_ids"]
        device = (self._cfg.device or "cpu").strip().lower()
        if device == "cuda":
            try:
                input_ids = input_ids.cuda()
            except Exception as e:  # pragma: no cover
                raise LLMUnavailableError(f"AirLLM: CUDA indisponível neste ambiente. Detalhe: {e}") from e
        else:
            input_ids = input_ids.cpu()

        out = model.generate(
            input_ids,
            max_new_tokens=max(32, int(self._cfg.max_new_tokens)),
            use_cache=True,
            return_dict_in_generate=True,
        )
        # Decodificar apenas os tokens novos ajuda o parser de JSON a não “se confundir”
        # com texto do prompt repetido.
        seq = out.sequences[0]
        try:
            gen_only = seq[input_ids.shape[-1] :]
            text = model.tokenizer.decode(gen_only)
        except Exception:
            text = model.tokenizer.decode(seq)
        return LLMResult(raw=text, provider=self.name)

