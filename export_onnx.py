"""Выгружает GigaAM-v3 (e2e_ctc) в ONNX: аудио 16 кГц -> log_probs.

Нужен один раз (требует torch). Результат — models/onnx/gigaam.onnx,
дальше приложение работает только через onnxruntime.
"""
import os
import sys
import time

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
CACHE = os.path.join(HERE, "models", "gigaam")
OUT_DIR = os.path.join(HERE, "models", "onnx")
OUT = os.path.join(OUT_DIR, "gigaam_e2e_ctc.onnx")

# GigaAM импортирует pyannote внутри функции VAD; transformers проверяет импорты
# по AST, поэтому нужна заглушка. Кэш HF держим рядом с проектом.
sys.path.append(os.path.join(HERE, "stubs"))
os.environ.setdefault("HF_HOME", os.path.join(HERE, "models", "hf"))
os.environ["HF_MODULES_CACHE"] = os.path.join(HERE, "models", "hf", "modules")
os.environ["HF_HUB_CACHE"] = os.path.join(HERE, "models", "hf", "hub")

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")


class Wrapper(torch.nn.Module):
    """Мел-признаки (1, 64, кадры) -> log_probs (1, кадры', классы).

    STFT не входит в граф: torch.onnx не умеет комплексные числа, поэтому
    мел-спектрограмма считается в features.py на numpy.
    """

    def __init__(self, asr):
        super().__init__()
        self.asr = asr

    def forward(self, features):
        length = torch.tensor([features.shape[-1]], dtype=torch.long)
        return self.asr.forward_for_export(features, length)


def main() -> int:
    from transformers import AutoModel

    os.makedirs(OUT_DIR, exist_ok=True)
    print("загрузка модели (torch)...", flush=True)
    model = AutoModel.from_pretrained(
        "ai-sage/GigaAM-v3",
        revision="e2e_ctc",
        trust_remote_code=True,
        cache_dir=CACHE,
    )
    asr = model.model.eval()
    wrapper = Wrapper(asr).eval()

    dummy = torch.zeros(1, 64, 99, dtype=torch.float32)
    with torch.no_grad():
        log_probs = wrapper(dummy)
    print("выход torch:", tuple(log_probs.shape), flush=True)

    print("экспорт в ONNX (может занять пару минут)...", flush=True)
    started = time.time()
    torch.onnx.export(
        wrapper,
        (dummy,),
        OUT,
        input_names=["features"],
        output_names=["log_probs"],
        dynamic_axes={"features": {2: "frames"}, "log_probs": {1: "frames_out"}},
        opset_version=17,
        do_constant_folding=True,
    )
    size_mb = os.path.getsize(OUT) / 1e6
    print(f"готово за {time.time() - started:.0f}s: {OUT} ({size_mb:.0f} МБ)", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
