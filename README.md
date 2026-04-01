## MonitorIA (OCR + IA local)

Este projeto captura a tela/janela ativa, faz OCR estruturado e (opcionalmente) usa um LLM local para:
- **limpar** o texto do OCR,
- **estruturar** perguntas + alternativas em JSON,
- e (opcionalmente) **sugerir** a alternativa mais provável.

Backend de LLM:
- **AirLLM-only** (via HuggingFace) — referência: [lyogavin/airllm](https://github.com/lyogavin/airllm)

## Rodando

### 1) Dependências

Instale as dependências do app (inclui AirLLM + torch):

```bash
pip install -r pythonIA/requirements.txt
```

Se você estiver em um ambiente sem GPU, mantenha `AIRLLM_DEVICE=cpu` (padrão).

### 2) Executar

```bash
python pythonIA/Python.py
```

Hotkeys:
- `Ctrl+Shift+S`: captura tela
- `Ctrl+Shift+W`: captura janela ativa
- `Ctrl+Shift+C`: alterna visibilidade da janela

## IA (AirLLM)

Na UI, marque “Refinar com IA local após captura (AirLLM)”.

### Variáveis de ambiente (LLM)

- **LLM_TWO_PASS**: `1` (padrão) usa modo “2-pass” (extrai estrutura primeiro, sugere depois). Ajuda quando OCR está muito ruidoso.
- **LLM_OCR_MAX_CHARS**: máximo de caracteres do OCR enviados ao LLM (padrão `9000`).

### AirLLM (Windows / CPU, recomendado)

Essas configs ficam em `pythonIA/airllm_client.py`.

- **AIRLLM_MODEL**: obrigatório (repo ID HF ou caminho local do modelo)
- **AIRLLM_DEVICE**: padrão `cpu`
- **AIRLLM_MAX_NEW_TOKENS**: padrão `512`
- **AIRLLM_MAX_INPUT_TOKENS**: padrão `3072`
- **AIRLLM_COMPRESSION**: `4bit` ou `8bit` (opcional)
- **AIRLLM_SHARDS_PATH**: diretório para salvar shards/layers (opcional)
- **AIRLLM_PROFILING**: `1` para profiling_mode (opcional)
- **HF_TOKEN** (ou `HUGGINGFACE_TOKEN`): para modelos gated (opcional)

## OCR / extração

- **TESSERACT_CMD**: caminho do `tesseract.exe` (se não detectar automaticamente)
- **TESSERACT_LANG**: padrão `eng+por`
- **TESSERACT_CONFIG**: padrão `--oem 3 --psm 3`
- **OCR_FAST**: `1` para pipeline mais leve (mais rápido, pode perder qualidade)

Recortes (para remover sidebar/taskbar antes do OCR):
- **OCR_CROP_LEFT_FRAC**, **OCR_CROP_RIGHT_FRAC**, **OCR_CROP_TOP_FRAC**, **OCR_CROP_BOTTOM_FRAC** (tela inteira)
- **OCR_CROP_WINDOW_LEFT_FRAC** ... (janela)

## UI Web (opcional)

- **WEB_ENABLE**: `1` inicia servidor web de resultados
- **WEB_PORT**: padrão `8765`
- **WEB_HISTORY**: padrão `25`
- **WEB_LOCAL_ONLY**: `1` restringe ao PC (127.0.0.1)

Na UI web, cada captura mostra:
- total de perguntas, quantas incompletas,
- backend usado,
- tempos (OCR+parse / LLM).

