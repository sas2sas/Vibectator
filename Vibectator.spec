# -*- mode: python ; coding: utf-8 -*-
# Сборка standalone-версии: onnxruntime + numpy + запись/трей, без torch.
import os

block_cipher = None
HERE = os.path.abspath(SPECPATH)

datas = [
    (os.path.join(HERE, "models", "onnx", "gigaam_int8.onnx"), os.path.join("models", "onnx")),
    (os.path.join(HERE, "models", "onnx", "tokenizer.model"), os.path.join("models", "onnx")),
    (os.path.join(HERE, "assets", "icon.png"), "assets"),
]

hiddenimports = [
    "pystray._win32",
    "PIL._tkinter_finder",
    "keyboard",
    "sounddevice",
    "onnxruntime",
    "sentencepiece",
    "features",
]

a = Analysis(
    [os.path.join(HERE, "dictate.py")],
    pathex=[HERE],
    binaries=[],
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    runtime_hooks=[],
    excludes=["torch", "transformers", "faster_whisper", "scipy", "matplotlib", "pandas"],
    cipher=block_cipher,
    noarchive=False,
)
pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name="Vibectator",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,
    console=False,
    icon=os.path.join(HERE, "assets", "icon.ico"),
    uac_admin=True,
    version=None,
)

coll = COLLECT(
    exe,
    a.binaries,
    a.zipfiles,
    a.datas,
    strip=False,
    upx=False,
    upx_exclude=[],
    name="Vibectator",
)
