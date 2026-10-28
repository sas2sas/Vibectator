"""Сравнивает torch-энкодер и ONNX на одном файле: логиты, текст, скорость.

Запуск: python verify_onnx.py <wav>
"""
import glob
import os
import sys
import time
import wave

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(HERE, "stubs"))
os.environ.setdefault("HF_HOME", os.path.join(HERE, "models", "hf"))
os.environ["HF_MODULES_CACHE"] = os.path.join(HERE, "models", "hf", "modules")
os.environ["HF_HUB_CACHE"] = os.path.join(HERE, "models", "hf", "hub")

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

ONNX_PATH = os.path.join(HERE, "models", "onnx", "gigaam_e2e_ctc.onnx")

import features as feat  # noqa: E402


def read_wav(path: str) -> np.ndarray:
    with wave.open(path, "rb") as fh:
        channels, rate = fh.getnchannels(), fh.getframerate()
        raw = fh.readframes(fh.getnframes())
    audio = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
    if channels > 1:
        audio = audio.reshape(-1, channels).mean(axis=1)
    if rate != 16000:
        idx = np.linspace(0, len(audio) - 1, int(len(audio) * 16000 / rate))
        audio = np.interp(idx, np.arange(len(audio)), audio).astype(np.float32)
    return audio


def find_tokenizer() -> str:
    for pattern in (
        os.path.join(HERE, "models", "hf", "hub", "**", "tokenizer.model"),
        os.path.join(HERE, "models", "gigaam", "**", "tokenizer.model"),
    ):
        found = glob.glob(pattern, recursive=True)
        if found:
            return found[0]
    raise FileNotFoundError("tokenizer.model не найден")


def decode_ctc(log_probs: np.ndarray, sp) -> str:
    ids = log_probs[0].argmax(axis=-1)
    blank = log_probs.shape[-1] - 1
    tokens, prev = [], -1
    for token in ids:
        token = int(token)
        if token != blank and token != prev:
            tokens.append(token)
        prev = token
    return sp.decode(tokens)


def main() -> int:
    if len(sys.argv) < 2:
        print(__doc__)
        return 2
    audio = read_wav(sys.argv[1])
    mel = feat.log_mel(audio)
    print(f"аудио: {len(audio) / 16000:.2f} с, признаки: {mel.shape}")

    import torch
    from sentencepiece import SentencePieceProcessor
    from transformers import AutoModel

    sp = SentencePieceProcessor(model_file=find_tokenizer())
    print(f"словарь: {sp.vocab_size()} токенов (+blank)")

    model = AutoModel.from_pretrained(
        "ai-sage/GigaAM-v3", revision="e2e_ctc", trust_remote_code=True,
        cache_dir=os.path.join(HERE, "models", "gigaam"),
    )
    asr = model.model.eval()

    t0 = time.time()
    with torch.no_grad():
        tensor = torch.from_numpy(mel)
        torch_logits = asr.forward_for_export(tensor, torch.tensor([tensor.shape[-1]])).numpy()
    t_torch = time.time() - t0

    import onnxruntime as ort

    session = ort.InferenceSession(ONNX_PATH, providers=["CPUExecutionProvider"])
    t1 = time.time()
    onnx_logits = session.run(None, {"features": mel})[0]
    t_onnx = time.time() - t1

    print(f"torch: {torch_logits.shape} за {t_torch * 1000:.0f} мс")
    print(f"onnx:  {onnx_logits.shape} за {t_onnx * 1000:.0f} мс")
    if torch_logits.shape == onnx_logits.shape:
        print(f"максимальное расхождение: {float(np.abs(torch_logits - onnx_logits).max()):.6f}")
    else:
        print("ВНИМАНИЕ: формы не совпадают")
    print(f"torch текст: {decode_ctc(torch_logits, sp)!r}")
    print(f"onnx  текст: {decode_ctc(onnx_logits, sp)!r}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
