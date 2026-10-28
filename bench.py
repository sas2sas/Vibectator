"""Замер скорости распознавания на одном WAV при разных настройках.

Запуск: python bench.py <wav> [язык]
Печатает время и RTF для комбинаций потоков CPU и beam_size.
"""
import os
import sys
import time
import wave

import numpy as np

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from faster_whisper import WhisperModel

HERE = os.path.dirname(os.path.abspath(__file__))
MODEL = os.path.join(HERE, "models", "large-v3-turbo")


def load_wav(path):
    with wave.open(path, "rb") as fh:
        params = fh.getparams()
        raw = fh.readframes(params.nframes)
    audio = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
    if params.nchannels > 1:
        audio = audio.reshape(-1, params.nchannels).mean(axis=1)
    return audio


def main():
    if len(sys.argv) < 2:
        print(__doc__)
        return 2
    path = sys.argv[1]
    lang = sys.argv[2] if len(sys.argv) > 2 else "ru"
    audio = load_wav(path)
    dur = len(audio) / 16000
    print(f"аудио: {dur:.1f}s, язык: {lang}, модель: large-v3-turbo int8")

    for threads in (8, 12, 16):
        model = WhisperModel(MODEL, device="cpu", compute_type="int8", cpu_threads=threads)
        for beam in (1, 2, 5):
            t0 = time.time()
            segs, _ = model.transcribe(
                audio,
                language=lang,
                beam_size=beam,
                vad_filter=True,
                condition_on_previous_text=False,
                without_timestamps=True,
                temperature=0.0,
            )
            text = " ".join(s.text.strip() for s in segs if s.text.strip())
            el = time.time() - t0
            print(f"threads={threads:2d} beam={beam}  {el:5.2f}s  RTF {el / dur:4.2f}  | {text[:80]}", flush=True)
        del model
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
