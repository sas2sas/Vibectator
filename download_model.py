"""Скачивает модель faster-whisper в локальный кэш и проверяет загрузку.

Если основной хост Hugging Face недоступен (например, отдаёт 401),
скрипт автоматически повторяет загрузку через зеркало hf-mirror.com.

Запуск: python download_model.py [имя_модели ...]
"""
import os
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(HERE, "models")
DEFAULT_MODELS = ["Systran/faster-whisper-large-v3-turbo"]
MIRROR = "https://hf-mirror.com"


def download(names: list[str], endpoint: str) -> None:
    os.environ["HF_ENDPOINT"] = endpoint
    os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"
    from faster_whisper import WhisperModel

    for name in names:
        print(f"[download] {name} via {endpoint}", flush=True)
        started = time.time()
        model = WhisperModel(
            name,
            device="cpu",
            compute_type="int8",
            download_root=MODELS_DIR,
            cpu_threads=max(1, (os.cpu_count() or 4) // 2),
        )
        del model
        print(f"[ok] {name} готово за {time.time() - started:.1f}s", flush=True)


def main() -> int:
    args = sys.argv[1:]
    endpoint = os.environ.get("HF_ENDPOINT", "https://huggingface.co")
    if "--mirror" in args:
        args = [a for a in args if a != "--mirror"]
        endpoint = MIRROR
    names = args or DEFAULT_MODELS
    try:
        download(names, endpoint)
        return 0
    except Exception as exc:  # noqa: BLE001
        if endpoint == MIRROR:
            print(f"[fail] зеркало тоже не сработало: {exc}", file=sys.stderr)
            return 1
        print(f"[warn] {endpoint} не сработал ({type(exc).__name__}); повтор через зеркало", file=sys.stderr)
    cmd = [sys.executable, os.path.abspath(__file__), "--mirror", *names]
    return subprocess.call(cmd)


if __name__ == "__main__":
    raise SystemExit(main())
