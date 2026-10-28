"""Скачивает CTranslate2-модель Whisper с ModelScope (когда Hugging Face недоступен).

Файлы кладутся в models/<короткое_имя>/ в том виде, который понимает WhisperModel(path).

Запуск: python download_ms.py [repo_id] [короткое_имя]
"""
import os
import sys

import requests

HERE = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(HERE, "models")

DEFAULT_REPO = "pengzhendong/faster-whisper-large-v3-turbo"
DEFAULT_NAME = "large-v3-turbo"

REQUIRED = ["config.json", "model.bin", "tokenizer.json", "preprocessor_config.json"]
OPTIONAL = ["vocabulary.json", "vocabulary.txt"]


def url_for(repo: str, path: str) -> str:
    return f"https://modelscope.cn/api/v1/models/{repo}/repo?Revision=master&FilePath={path}"


def fetch(repo: str, path: str, dest: str) -> bool:
    tmp = dest + ".part"
    have = os.path.getsize(tmp) if os.path.exists(tmp) else 0
    headers = {"Range": f"bytes={have}-"} if have else {}
    with requests.get(url_for(repo, path), stream=True, headers=headers, timeout=60) as resp:
        if resp.status_code == 404:
            return False
        resp.raise_for_status()
        mode = "ab" if have and resp.status_code == 206 else "wb"
        if mode == "wb":
            have = 0
        total = int(resp.headers.get("Content-Length", 0)) + have
        done = have
        step = max(1, total // 20)
        with open(tmp, mode) as fh:
            for chunk in resp.iter_content(chunk_size=1 << 20):
                if not chunk:
                    continue
                fh.write(chunk)
                done += len(chunk)
                if total and done // step != (done - len(chunk)) // step:
                    print(f"  {path}: {done / 1e6:.0f}/{total / 1e6:.0f} MB", flush=True)
    os.replace(tmp, dest)
    print(f"  {path}: {os.path.getsize(dest) / 1e6:.1f} MB ok", flush=True)
    return True


def main() -> int:
    repo = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_REPO
    name = sys.argv[2] if len(sys.argv) > 2 else DEFAULT_NAME
    target = os.path.join(MODELS_DIR, name)
    os.makedirs(target, exist_ok=True)
    print(f"[modelscope] {repo} -> {target}", flush=True)
    for path in REQUIRED:
        if not fetch(repo, path, os.path.join(target, path)):
            print(f"[fail] нет файла {path}", file=sys.stderr)
            return 1
    for path in OPTIONAL:
        fetch(repo, path, os.path.join(target, path))
    print("[ok] модель на месте", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
