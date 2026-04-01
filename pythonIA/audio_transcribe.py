from __future__ import annotations

import os
import time


def _env_int(name: str, default: int) -> int:
    try:
        return int(os.environ.get(name, str(default)).strip())
    except Exception:
        return default


def record_system_audio_wasapi(seconds: int = 15, samplerate: int = 16000):
    """
    Captura áudio do PC (loopback WASAPI no Windows) e retorna numpy float32 mono.
    Requer: sounddevice
    """
    import numpy as np
    import sounddevice as sd
    import inspect

    # saída padrão do Windows em loopback
    dev = sd.default.device
    if isinstance(dev, (list, tuple)) and len(dev) >= 2:
        out_dev = dev[1]
    else:
        out_dev = None

    # Compatibilidade entre versões:
    # - sounddevice/PortAudio mais novos suportam WasapiSettings(loopback=True)
    # - alguns builds não expõem esse argumento -> oferecer fallback via biblioteca soundcard (WASAPI)
    try:
        sig = inspect.signature(sd.WasapiSettings)
        supports_loopback = "loopback" in sig.parameters
    except Exception:
        supports_loopback = False

    if not supports_loopback:
        raise TypeError("WasapiSettings(loopback=True) não suportado nesta versão/build do sounddevice.")

    with sd.InputStream(
        samplerate=samplerate,
        channels=1,
        dtype="float32",
        device=out_dev,
        blocksize=0,
        latency="low",
        extra_settings=sd.WasapiSettings(loopback=True),
    ) as stream:
        frames = []
        t_end = time.time() + seconds
        while time.time() < t_end:
            data, _ = stream.read(1024)
            frames.append(data.copy())
    audio = np.concatenate(frames, axis=0).reshape(-1)
    return audio, samplerate


def record_system_audio_soundcard(seconds: int = 15, samplerate: int = 16000):
    """
    Fallback: captura loopback via WASAPI usando a lib 'soundcard' (Windows).
    Requer: soundcard
    """
    import numpy as np
    import soundcard as sc

    speaker = sc.default_speaker()
    with speaker.recorder(samplerate=samplerate) as rec:
        data = rec.record(numframes=int(seconds * samplerate))
    # soundcard retorna (frames, channels)
    if data.ndim == 2 and data.shape[1] >= 1:
        mono = data[:, 0]
    else:
        mono = data.reshape(-1)
    mono = mono.astype(np.float32, copy=False)
    return mono, samplerate


def record_system_audio(seconds: int = 15, samplerate: int = 16000):
    """Tenta sounddevice(WASAPI loopback) e cai para soundcard se necessário."""
    try:
        return record_system_audio_wasapi(seconds=seconds, samplerate=samplerate)
    except Exception:
        return record_system_audio_soundcard(seconds=seconds, samplerate=samplerate)


def transcribe_faster_whisper(audio, samplerate: int) -> str:
    """
    Transcreve áudio para texto (somente transcrição).
    Requer: faster-whisper
    """
    from faster_whisper import WhisperModel

    model_size = os.environ.get("STT_MODEL", "small").strip()  # tiny|base|small|medium|large-v3
    device = os.environ.get("STT_DEVICE", "cpu").strip()
    compute_type = os.environ.get("STT_COMPUTE_TYPE", "int8").strip()

    model = WhisperModel(model_size, device=device, compute_type=compute_type)
    segments, _info = model.transcribe(audio, language=os.environ.get("STT_LANG", "").strip() or None)
    parts = []
    for seg in segments:
        parts.append(seg.text.strip())
    return " ".join(p for p in parts if p)


def main():
    seconds = _env_int("STT_SECONDS", 15)
    sr = _env_int("STT_SR", 16000)
    audio, samplerate = record_system_audio(seconds=seconds, samplerate=sr)
    text = transcribe_faster_whisper(audio, samplerate)
    print(text)


if __name__ == "__main__":
    main()

