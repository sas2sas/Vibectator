"""Проверка распознавания на готовом файле: python test_asr.py <файл> [язык]

Печатает текст, время распознавания и RTF (во сколько раз быстрее реального времени).
Любой формат, который декодирует PyAV (wav, mp3, ogg, m4a...).
"""
import os
import sys
import time

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

from faster_whisper import WhisperModel

HERE = os.path.dirname(os.path.abspath(__file__))
MODEL = os.path.join(HERE, "models", "large-v3-turbo")


def main() -> int:
    if len(sys.argv) < 2:
        print(__doc__)
        return 2
    path = sys.argv[1]
    lang = sys.argv[2] if len(sys.argv) > 2 else "ru"

    t0 = time.time()
    model = WhisperModel(MODEL, device="cpu", compute_type="int8", cpu_threads=8)
    print(f"модель загружена за {time.time() - t0:.1f}s", flush=True)

    t1 = time.time()
    segments, info = model.transcribe(
        path,
        language=lang,
        beam_size=5,
        vad_filter=True,
        condition_on_previous_text=False,
        initial_prompt="Это диктовка на русском языке. Расставляй знаки препинания и заглавные буквы.",
        temperature=0.0,
    )
    text = " ".join(s.text.strip() for s in segments if s.text.strip())
    elapsed = time.time() - t1
    print(f"длительность аудио: {info.duration:.1f}s")
    print(f"распознавание: {elapsed:.2f}s  (RTF {elapsed / max(info.duration, 0.01):.2f})")
    print(f"текст: {text}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
