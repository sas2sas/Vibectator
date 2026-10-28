"""Проверка GigaAM через общий движок из dictate.py: python test_gigaam.py <wav>"""
import os
import sys
import time
import wave

import numpy as np

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import dictate  # noqa: E402


def read_wav(path: str) -> np.ndarray:
    with wave.open(path, "rb") as fh:
        raw = fh.readframes(fh.getnframes())
        ch = fh.getnchannels()
    audio = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
    if ch > 1:
        audio = audio.reshape(-1, ch).mean(axis=1)
    return audio


def main() -> int:
    if len(sys.argv) < 2:
        print(__doc__)
        return 2
    path = sys.argv[1]
    if len(sys.argv) > 2:
        os.environ["DICTATE_REVISION"] = sys.argv[2]
    cfg = dictate.load_config()
    cfg["engine"] = "gigaam"
    if os.environ.get("DICTATE_REVISION"):
        cfg["gigaam_revision"] = os.environ["DICTATE_REVISION"]

    audio = read_wav(path)
    duration = len(audio) / dictate.SAMPLE_RATE

    t0 = time.time()
    engine = dictate.build_engine(cfg)
    load = time.time() - t0

    text, elapsed = engine.transcribe(audio)
    print(f"загрузка: {load:.1f}s | аудио {duration:.1f}s | распознавание {elapsed:.2f}s | RTF {elapsed / duration:.3f}")
    print(f"текст: {text}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
