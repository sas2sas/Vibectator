"""Квантует ONNX-модель GigaAM в int8: 886 МБ -> ~250 МБ.

Запуск: python quantize_onnx.py
"""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(HERE, "models", "onnx", "try_torchscript.onnx")
DST = os.path.join(HERE, "models", "onnx", "gigaam_int8.onnx")

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")


def main() -> int:
    from onnxruntime.quantization import QuantType, quantize_dynamic

    if not os.path.exists(SRC):
        print(f"нет файла {SRC}", file=sys.stderr)
        return 2
    print(f"квантование {os.path.getsize(SRC) / 1e6:.0f} МБ -> int8...", flush=True)
    quantize_dynamic(
        model_input=SRC,
        model_output=DST,
        weight_type=QuantType.QInt8,
        extra_options={"MatMulConstBOnly": False},
    )
    print(f"готово: {DST} ({os.path.getsize(DST) / 1e6:.0f} МБ)", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
