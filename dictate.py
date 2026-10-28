"""Локальный голосовой ввод для Windows: хоткей -> запись -> распознавание -> вставка.

Движок: faster-whisper (CTranslate2, CPU int8). После скачивания модели работает офлайн.

Использование:
    python dictate.py                # запуск в трее
    python dictate.py --list-devices # список микрофонов
    python dictate.py --wav file.wav # распознать готовый файл и выйти
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import queue
import sys
import threading
import time
import wave

import numpy as np

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

HERE = os.path.dirname(os.path.abspath(__file__))
# В собранном виде: HERE — каталог ресурсов (_internal), APP_DIR — рядом с exe.
APP_DIR = HERE
FROZEN = bool(getattr(sys, "frozen", False))
if FROZEN:
    APP_DIR = os.path.dirname(os.path.abspath(sys.executable))


def _pick_data_dir() -> str:
    """Первая папка из списка, куда реально можно писать."""
    if not FROZEN:
        return HERE
    candidates = []
    for env in ("LOCALAPPDATA", "APPDATA"):
        base = os.environ.get(env)
        if base:
            candidates.append(os.path.join(base, "Vibectator"))
    candidates.append(os.path.join(APP_DIR, "data"))
    temp = os.environ.get("TEMP") or os.environ.get("TMP")
    if temp:
        candidates.append(os.path.join(temp, "Vibectator"))
    candidates.append(HERE)

    for path in candidates:
        try:
            os.makedirs(path, exist_ok=True)
            probe = os.path.join(path, ".write_probe")
            with open(probe, "w", encoding="utf-8") as fh:
                fh.write("ok")
            os.remove(probe)
            return path
        except Exception:  # noqa: BLE001
            continue
    return HERE


DATA_DIR = _pick_data_dir()


def _migrate_old_data() -> None:
    """Переносит настройки из прежней папки «GigaAM Dictation», если она есть."""
    if not FROZEN:
        return
    parent = os.path.dirname(DATA_DIR)
    old = os.path.join(parent, "GigaAM Dictation")
    if not os.path.isdir(old) or os.path.abspath(old) == os.path.abspath(DATA_DIR):
        return
    import shutil

    for name in ("config.json", "last_text.txt"):
        src = os.path.join(old, name)
        dst = os.path.join(DATA_DIR, name)
        if os.path.exists(src) and not os.path.exists(dst):
            try:
                shutil.copy2(src, dst)
            except Exception:  # noqa: BLE001
                pass


_migrate_old_data()

CONFIG_PATH = os.path.join(DATA_DIR, "config.json")
LOG_PATH = os.path.join(DATA_DIR, "dictation.log")
LAST_TEXT_PATH = os.path.join(DATA_DIR, "last_text.txt")
MODELS_DIR = os.path.join(APP_DIR, "models")
SAMPLE_RATE = 16000

DEFAULTS = {
    "engine": "onnx",              # onnx (по умолчанию) | gigaam (torch) | whisper
    "onnx_model": "models/onnx/gigaam_int8.onnx",
    "tokenizer": "models/onnx/tokenizer.model",
    "gigaam_revision": "e2e_ctc",  # e2e_ctc | e2e_rnnt | ctc | rnnt
    "gigaam_cache": "models/gigaam",
    "model": "Systran/faster-whisper-large-v3-turbo",
    "language": "ru",
    "hotkey": "tilde",
    "mode": "hold",                # hold (держать) | toggle (переключать)
    "compute_type": "int8",
    "cpu_threads": 8,
    "beam_size": 1,
    "vad_filter": True,
    "initial_prompt": "Это диктовка на русском языке. Расставляй знаки препинания и заглавные буквы.",
    "input_device": None,          # None = системный по умолчанию
    "insert_method": "type",       # type (печать) | paste (Ctrl+V) | clipboard (только копировать)
    "trailing_space": True,
    "restore_clipboard": True,
    "beep": True,
    "tray": True,                  # False — работать без иконки в трее
    "vad": True,                   # отсекать записи без речи
    "min_speech_seconds": 0.3,
    "min_seconds": 0.35,
    "max_seconds": 60,             # предохранитель от зависшей записи
}
COMBO = [k.strip().lower() for k in DEFAULTS["hotkey"].split("+")]

log = logging.getLogger("dictate")


def load_config() -> dict:
    cfg = dict(DEFAULTS)
    if os.path.exists(CONFIG_PATH):
        try:
            with open(CONFIG_PATH, "r", encoding="utf-8-sig") as fh:
                cfg.update(json.load(fh))
        except Exception as exc:  # noqa: BLE001
            log.warning("config.json не прочитан (%s), беру значения по умолчанию", exc)
    else:
        with open(CONFIG_PATH, "w", encoding="utf-8") as fh:
            json.dump(cfg, fh, ensure_ascii=False, indent=2)
    return cfg


def beep(kind: str) -> None:
    """Короткие сигналы без внешних зависимостей."""
    try:
        import winsound

        table = {
            "start": ((880, 70),),
            "stop": ((660, 60),),
            "done": ((1046, 70), (1318, 70)),
            "error": ((320, 160),),
            "cancel": ((440, 60), (330, 60)),
        }
        for freq, dur in table.get(kind, ()):
            winsound.Beep(freq, dur)
    except Exception:  # noqa: BLE001
        pass


class Recorder:
    """Непрерывная запись с микрофона в память, 16 кГц моно float32."""

    def __init__(self, device=None):
        import sounddevice as sd

        self._sd = sd
        self.device = device
        self._stream = None
        self._chunks: list[np.ndarray] = []
        self._lock = threading.Lock()
        self._recording = False

    def start(self) -> None:
        self._chunks = []
        self._recording = True

        def callback(indata, frames, time_info, status):  # noqa: ANN001, ARG001
            if status:
                log.debug("audio status: %s", status)
            with self._lock:
                if self._recording:
                    self._chunks.append(indata.copy())

        self._stream = self._sd.InputStream(
            samplerate=SAMPLE_RATE,
            channels=1,
            dtype="float32",
            device=self.device,
            callback=callback,
            blocksize=0,
        )
        self._stream.start()

    def stop(self) -> np.ndarray:
        self._recording = False
        if self._stream is not None:
            stream, self._stream = self._stream, None
            for action in (stream.stop, stream.close):
                try:
                    action()
                except Exception:  # noqa: BLE001
                    log.exception("ошибка при остановке аудиопотока")
        with self._lock:
            if not self._chunks:
                return np.zeros(0, dtype=np.float32)
            audio = np.concatenate(self._chunks, axis=0).reshape(-1)
            self._chunks = []
        return audio

    def cancel(self) -> None:
        self._recording = False
        if self._stream is not None:
            stream, self._stream = self._stream, None
            for action in (stream.stop, stream.close):
                try:
                    action()
                except Exception:  # noqa: BLE001
                    log.exception("ошибка при отмене аудиопотока")
        with self._lock:
            self._chunks = []

    @property
    def active(self) -> bool:
        return self._recording


def _resolve_path(value: str) -> str:
    if os.path.isabs(value) or not value:
        return value
    for base in (HERE, MODELS_DIR):
        candidate = os.path.join(base, value)
        if os.path.exists(candidate):
            return candidate
    return os.path.join(HERE, value)


class WhisperEngine:
    """faster-whisper: тёплая модель, загружается один раз."""

    def __init__(self, cfg: dict):
        from faster_whisper import WhisperModel

        self.cfg = cfg
        model_ref = cfg["model"]
        if not os.path.isabs(model_ref):
            for base in (HERE, MODELS_DIR):
                candidate = os.path.join(base, model_ref)
                if os.path.isdir(candidate):
                    model_ref = candidate
                    break
        if not (os.path.isdir(model_ref) or "/" in str(model_ref)):
            raise SystemExit(
                f"Модель не найдена: {model_ref}\n"
                "Скачайте её: python download_ms.py"
            )
        self.model = WhisperModel(
            model_ref,
            device="cpu",
            compute_type=cfg["compute_type"],
            cpu_threads=max(1, int(cfg["cpu_threads"])),
            download_root=MODELS_DIR,
            num_workers=1,
        )
        self.warmup()

    def warmup(self) -> None:
        """Прогрев на тишине — первый вызов всегда медленнее."""
        try:
            audio = np.zeros(SAMPLE_RATE, dtype=np.float32)
            segments, _ = self.model.transcribe(audio, language=self.cfg["language"], beam_size=1)
            list(segments)
        except Exception as exc:  # noqa: BLE001
            log.debug("warmup: %s", exc)

    def transcribe(self, audio: np.ndarray) -> tuple[str, float]:
        started = time.time()
        segments, info = self.model.transcribe(
            audio,
            language=self.cfg["language"],
            beam_size=int(self.cfg["beam_size"]),
            vad_filter=bool(self.cfg["vad_filter"]),
            vad_parameters={"min_silence_duration_ms": 300, "speech_pad_ms": 120},
            condition_on_previous_text=False,
            without_timestamps=True,
            initial_prompt=self.cfg.get("initial_prompt") or None,
            temperature=0.0,
        )
        text = " ".join(seg.text.strip() for seg in segments if seg.text.strip())
        return text, time.time() - started


class GigaAMEngine:
    """GigaAM-v3 (русский ASR): точнее и кратно быстрее Whisper.

    Модель берётся из transformers с trust_remote_code; её загрузчик аудио
    вызывает внешний ffmpeg, поэтому мы подменяем его чтением WAV напрямую.
    """

    def __init__(self, cfg: dict):
        # GigaAM импортирует pyannote только для длинных записей; заглушка
        # нужна, чтобы transformers не отказался грузить trust_remote_code.
        stubs = os.path.join(HERE, "stubs")
        if os.path.isdir(stubs) and stubs not in sys.path:
            sys.path.append(stubs)

        # Весь кэш Hugging Face держим рядом с программой: запись в профиль
        # пользователя может быть запрещена политикой.
        hf_home = _resolve_path(cfg.get("hf_home", "models/hf"))
        os.makedirs(hf_home, exist_ok=True)
        os.environ["HF_HOME"] = hf_home
        os.environ["HF_MODULES_CACHE"] = os.path.join(hf_home, "modules")
        os.environ["HF_HUB_CACHE"] = os.path.join(hf_home, "hub")
        os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")

        import torch
        from transformers import AutoModel

        self.cfg = cfg
        self._torch = torch
        torch.set_num_threads(max(1, int(cfg["cpu_threads"])))
        revision = cfg.get("gigaam_revision", "e2e_ctc")
        cache = _resolve_path(cfg.get("gigaam_cache", "models/gigaam"))
        log.info("загрузка GigaAM-v3 [%s]", revision)
        print("загрузка модели GigaAM (обычно 10-20 секунд)...", flush=True)
        self.model = AutoModel.from_pretrained(
            "ai-sage/GigaAM-v3",
            revision=revision,
            trust_remote_code=True,
            cache_dir=cache,
        )
        self.model.eval()
        self._patch_audio_loader()
        print("веса прочитаны, прогрев...", flush=True)
        self.warmup()
        print("модель готова", flush=True)

    def _patch_audio_loader(self) -> None:
        torch = self._torch

        def reader(path: str, sample_rate: int = SAMPLE_RATE):  # noqa: ARG001
            with wave.open(path, "rb") as fh:
                raw = fh.readframes(fh.getnframes())
            audio = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
            return torch.from_numpy(audio)

        # load_audio вызывается из методов модели. hydra может импортировать
        # модуль модели повторно (под другим именем), поэтому патчим все копии.
        patched = 0
        for name, mod in list(sys.modules.items()):
            if mod is None or "gigaam" not in name.lower():
                continue
            namespace = getattr(mod, "__dict__", None)
            if isinstance(namespace, dict) and callable(namespace.get("load_audio")):
                namespace["load_audio"] = reader
                patched += 1
        if patched:
            log.info("загрузчик аудио подменён в %d модулях (ffmpeg не нужен)", patched)
        else:
            log.warning("не нашёл load_audio — понадобится ffmpeg в PATH")

    def _write_wav(self, audio: np.ndarray) -> str:
        import tempfile

        fd, path = tempfile.mkstemp(suffix=".wav", prefix="dictate_")
        os.close(fd)
        with wave.open(path, "wb") as fh:
            fh.setnchannels(1)
            fh.setsampwidth(2)
            fh.setframerate(SAMPLE_RATE)
            fh.writeframes((np.clip(audio, -1.0, 1.0) * 32767).astype(np.int16).tobytes())
        return path

    def warmup(self) -> None:
        try:
            with self._torch.inference_mode():
                self._run(np.zeros(SAMPLE_RATE // 2, dtype=np.float32))
        except Exception as exc:  # noqa: BLE001
            log.debug("warmup: %s", exc)

    def _run(self, audio: np.ndarray) -> str:
        # модель не принимает записи длиннее ~25 с — режем сами
        window = SAMPLE_RATE * 20
        if len(audio) <= window:
            return self._one(audio)
        parts = []
        for start in range(0, len(audio), window):
            piece = audio[start:start + window]
            if len(piece) < SAMPLE_RATE * 0.2:
                continue
            parts.append(self._one(piece))
        return " ".join(part for part in parts if part)

    def _one(self, audio: np.ndarray) -> str:
        path = self._write_wav(audio)
        try:
            return self.model.transcribe(path)
        finally:
            try:
                os.remove(path)
            except OSError:
                pass

    def transcribe(self, audio: np.ndarray) -> tuple[str, float]:
        started = time.time()
        with self._torch.inference_mode():
            text = self._run(audio)
        return text, time.time() - started


class OnnxEngine:
    """GigaAM-v3 в ONNX: без torch, старт ~2 с, модель 225 МБ (int8).

    Мел-признаки считаются в features.py (STFT в ONNX не экспортируется),
    декодирование — обычный CTC + SentencePiece.
    """

    def __init__(self, cfg: dict):
        import onnxruntime as ort
        from sentencepiece import SentencePieceProcessor

        self.cfg = cfg
        model_path = _resolve_path(cfg.get("onnx_model", "models/onnx/gigaam_int8.onnx"))
        tok_path = _resolve_path(cfg.get("tokenizer", "models/onnx/tokenizer.model"))
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"ONNX-модель не найдена: {model_path}")
        if not os.path.exists(tok_path):
            raise FileNotFoundError(f"токенизатор не найден: {tok_path}")

        opts = ort.SessionOptions()
        opts.intra_op_num_threads = max(1, int(cfg.get("cpu_threads", 8)))
        opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        self.session = ort.InferenceSession(
            model_path, sess_options=opts, providers=["CPUExecutionProvider"]
        )
        self.sp = SentencePieceProcessor(model_file=tok_path)
        self.blank = None
        log.info("ONNX-модель загружена: %s", os.path.basename(model_path))

    def _decode(self, log_probs: np.ndarray) -> str:
        ids = log_probs[0].argmax(axis=-1)
        if self.blank is None:
            self.blank = log_probs.shape[-1] - 1
        tokens, prev = [], -1
        for token in ids:
            token = int(token)
            if token != self.blank and token != prev:
                tokens.append(token)
            prev = token
        return self.sp.decode(tokens) if tokens else ""

    def transcribe(self, audio: np.ndarray) -> tuple[str, float]:
        import features

        started = time.time()
        if audio.size < 1600:  # меньше 0,1 с
            return "", time.time() - started
        mel = features.log_mel(audio)
        inputs = {
            "features": mel,
            "lengths": np.array([mel.shape[-1]], dtype=np.int64),
        }
        log_probs = self.session.run(None, inputs)[0]
        return self._decode(log_probs), time.time() - started


def build_engine(cfg: dict):
    engine = str(cfg.get("engine", "onnx")).lower()
    if engine == "onnx":
        try:
            return OnnxEngine(cfg)
        except Exception as exc:  # noqa: BLE001
            log.exception("ONNX-движок не запустился")
            print(f"ONNX недоступен ({exc})", file=sys.stderr)
    if engine in ("onnx", "gigaam"):
        try:
            return GigaAMEngine(cfg)
        except Exception as exc:  # noqa: BLE001
            log.exception("GigaAM (torch) не запустился")
            print(f"GigaAM недоступен ({exc})", file=sys.stderr)
    return WhisperEngine(cfg)


def format_text(text: str, cfg: dict) -> str:
    text = " ".join(text.split())
    if not text:
        return ""
    text = text[0].upper() + text[1:]
    if cfg["trailing_space"] and not text.endswith((" ", "\n")):
        text += " "
    return text


def _type_unicode(text: str) -> None:
    """Печатает текст через SendInput с KEYEVENTF_UNICODE — работает для кириллицы.

    Важно: INPUT — это union с MOUSEINPUT, поэтому на x64 его размер 40 байт,
    а не 4 + sizeof(KEYBDINPUT). С «короткой» структурой SendInput отклоняется
    с ERROR_INVALID_PARAMETER, и печать молча не происходит.
    """
    import ctypes
    from ctypes import wintypes

    KEYEVENTF_UNICODE = 0x0004
    KEYEVENTF_KEYUP = 0x0002
    INPUT_KEYBOARD = 1

    class KEYBDINPUT(ctypes.Structure):
        _fields_ = [
            ("wVk", wintypes.WORD),
            ("wScan", wintypes.WORD),
            ("dwFlags", wintypes.DWORD),
            ("time", wintypes.DWORD),
            ("dwExtraInfo", ctypes.POINTER(ctypes.c_ulong)),
        ]

    class MOUSEINPUT(ctypes.Structure):
        _fields_ = [
            ("dx", wintypes.LONG),
            ("dy", wintypes.LONG),
            ("mouseData", wintypes.DWORD),
            ("dwFlags", wintypes.DWORD),
            ("time", wintypes.DWORD),
            ("dwExtraInfo", ctypes.POINTER(ctypes.c_ulong)),
        ]

    class _INPUTUNION(ctypes.Union):
        _fields_ = [("ki", KEYBDINPUT), ("mi", MOUSEINPUT)]

    class INPUT(ctypes.Structure):
        _anonymous_ = ("u",)
        _fields_ = [("type", wintypes.DWORD), ("u", _INPUTUNION)]

    user32 = ctypes.WinDLL("user32", use_last_error=True)
    user32.SendInput.argtypes = [wintypes.UINT, ctypes.c_void_p, ctypes.c_int]
    user32.SendInput.restype = wintypes.UINT
    size = ctypes.sizeof(INPUT)

    sent = 0
    for char in text:
        for flags in (KEYEVENTF_UNICODE, KEYEVENTF_UNICODE | KEYEVENTF_KEYUP):
            event = INPUT(
                type=INPUT_KEYBOARD,
                ki=KEYBDINPUT(0, ord(char), flags, 0, None),
            )
            if user32.SendInput(1, ctypes.byref(event), size) != 1:
                log.error(
                    "SendInput не сработал (код %s) при печати символа %r",
                    ctypes.get_last_error(),
                    char,
                )
                return
            sent += 1
    log.debug("напечатано %d символов через SendInput", sent)
    return None


def _send_paste() -> None:
    """Ctrl+V с человеческими паузами: некоторые окна не ловят «мгновенное» нажатие."""
    import keyboard

    keyboard.press("ctrl")
    time.sleep(0.03)
    keyboard.press("v")
    time.sleep(0.06)
    keyboard.release("v")
    time.sleep(0.03)
    keyboard.release("ctrl")


def _foreground_title() -> str:
    """Заголовок окна, которое сейчас в фокусе — куда уйдёт вставка."""
    try:
        import ctypes

        user32 = ctypes.windll.user32
        hwnd = user32.GetForegroundWindow()
        length = user32.GetWindowTextLengthW(hwnd)
        buf = ctypes.create_unicode_buffer(length + 1)
        user32.GetWindowTextW(hwnd, buf, length + 1)
        return buf.value or "<без заголовка>"
    except Exception:  # noqa: BLE001
        return "<неизвестно>"


def insert_text(text: str, cfg: dict) -> None:
    import pyperclip

    method = cfg["insert_method"]
    log.info("вставка (%s) в окно «%s»: %r", method, _foreground_title(), text[:60])

    # страховка: текст всегда попадает в буфер обмена.
    # Если автопечать заблокирована политикой, его можно вставить вручную (Ctrl+V).
    try:
        pyperclip.copy(text)
        log.info("текст скопирован в буфер обмена")
    except Exception:  # noqa: BLE001
        log.warning("не удалось скопировать текст в буфер обмена")

    try:
        with open(LAST_TEXT_PATH, "w", encoding="utf-8") as fh:
            fh.write(text)
    except Exception:  # noqa: BLE001
        log.debug("не удалось записать last_text.txt")

    if method == "clipboard":
        pyperclip.copy(text)
        return

    if method == "type":
        _type_unicode(text)
        return

    previous = None
    if cfg["restore_clipboard"]:
        try:
            previous = pyperclip.paste()
        except Exception:  # noqa: BLE001
            log.debug("не удалось прочитать буфер обмена")
    pyperclip.copy(text)
    time.sleep(0.06)
    if pyperclip.paste() != text:
        log.warning("буфер обмена не принял текст, печатаю напрямую")
        _type_unicode(text)
        return
    _send_paste()
    if cfg["restore_clipboard"] and previous is not None:
        time.sleep(0.4)
        try:
            pyperclip.copy(previous)
        except Exception:  # noqa: BLE001
            pass


_VK_NAMES = {
    "ctrl": 0x11,
    "control": 0x11,
    "alt": 0x12,
    "shift": 0x10,
    "win": 0x5B,
    "windows": 0x5B,
    "space": 0x20,
    "esc": 0x1B,
    "escape": 0x1B,
    "tab": 0x09,
    "enter": 0x0D,
    "return": 0x0D,
    "pause": 0x13,
    "insert": 0x2D,
    "capslock": 0x14,
    "caps": 0x14,
    "tilde": 0xC0,
    "backtick": 0xC0,
    "oem3": 0xC0,
    "ё": 0xC0,
    "numlock": 0x90,
    "scrolllock": 0x91,
    "apps": 0x5D,
    "home": 0x24,
    "end": 0x23,
}
for _i in range(1, 13):
    _VK_NAMES[f"f{_i}"] = 0x6F + _i
_MODIFIER_VKS = {0x10, 0x11, 0x12, 0x5B, 0x5C}
_SWALLOW_VKS = {0x14, 0x90, 0x91, 0xC0}  # CapsLock, NumLock, ScrollLock, Ё/~


def _vk(name: str):
    name = name.strip().lower()
    if name in _VK_NAMES:
        return _VK_NAMES[name]
    if len(name) == 1 and (name.isalpha() or name.isdigit()):
        return ord(name.upper())
    return None


def _key_down(name: str) -> bool:
    """Физически ли нажата клавиша прямо сейчас (бывает недоступно в удалённых сессиях)."""
    import ctypes

    vk = _vk(name)
    if vk is None:
        return False
    return bool(ctypes.windll.user32.GetAsyncKeyState(vk) & 0x8000)


class HotkeyHook:
    """Низкоуровневый хук клавиатуры: ловит и нажатие, и отпускание одной клавиши.

    Опрос GetAsyncKeyState на части систем (удалённые сессии, некоторый RDP)
    не видит клавиш, поэтому используем WH_KEYBOARD_LL.
    """

    WH_KEYBOARD_LL = 13
    WM_KEYDOWN, WM_KEYUP = 0x0100, 0x0101
    WM_SYSKEYDOWN, WM_SYSKEYUP = 0x0104, 0x0105

    def __init__(self, combo, on_press, on_release, on_cancel=None):
        self.required = [vk for vk in (_vk(k) for k in combo) if vk is not None]
        if not self.required:
            raise ValueError(f"не понимаю комбинацию: {combo}")
        self.trigger = self.required[-1]
        self.modifiers = set(self.required[:-1])
        self.on_press = on_press
        self.on_release = on_release
        self.on_cancel = on_cancel
        self._held_modifiers: set[int] = set()
        self._callback = None
        self._hook = None
        self.pressed_any = False

    def start(self) -> None:
        threading.Thread(target=self._run, daemon=True).start()

    def _fire(self, func) -> None:
        if func is None:
            return
        try:
            threading.Thread(target=func, daemon=True).start()
        except Exception:  # noqa: BLE001
            log.exception("не удалось запустить обработчик хоткея")

    def _run(self) -> None:
        import ctypes
        from ctypes import wintypes

        user32 = ctypes.WinDLL("user32", use_last_error=True)
        kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
        HOOKPROC = ctypes.WINFUNCTYPE(
            ctypes.c_ssize_t, ctypes.c_int, wintypes.WPARAM, wintypes.LPARAM
        )

        class KBDLLHOOKSTRUCT(ctypes.Structure):
            _fields_ = [
                ("vkCode", wintypes.DWORD),
                ("scanCode", wintypes.DWORD),
                ("flags", wintypes.DWORD),
                ("time", wintypes.DWORD),
                ("dwExtraInfo", ctypes.POINTER(ctypes.c_ulong)),
            ]

        def proc(n_code, w_param, l_param):  # noqa: ANN001
            swallow = False
            try:
                if n_code == 0:
                    data = ctypes.cast(
                        l_param, ctypes.POINTER(KBDLLHOOKSTRUCT)
                    ).contents
                    swallow = self._handle(data.vkCode, w_param)
            except Exception:  # noqa: BLE001
                log.exception("ошибка в обработчике клавиатуры")
            if swallow:
                return 1  # гасим клавишу: CapsLock не переключит регистр
            return user32.CallNextHookEx(None, n_code, w_param, l_param)

        self._callback = HOOKPROC(proc)

        # Без явных типов 64-битный HMODULE обрезается до 32 бит и хук не ставится.
        kernel32.GetModuleHandleW.restype = wintypes.HMODULE
        kernel32.GetModuleHandleW.argtypes = [wintypes.LPCWSTR]
        user32.SetWindowsHookExW.restype = wintypes.HHOOK
        user32.SetWindowsHookExW.argtypes = [
            ctypes.c_int,
            HOOKPROC,
            wintypes.HINSTANCE,
            wintypes.DWORD,
        ]
        user32.CallNextHookEx.restype = ctypes.c_ssize_t
        user32.CallNextHookEx.argtypes = [
            wintypes.HHOOK,
            ctypes.c_int,
            wintypes.WPARAM,
            wintypes.LPARAM,
        ]

        module = kernel32.GetModuleHandleW(None)
        self._hook = user32.SetWindowsHookExW(
            self.WH_KEYBOARD_LL, self._callback, module, 0
        )
        if not self._hook:
            self._hook = user32.SetWindowsHookExW(
                self.WH_KEYBOARD_LL, self._callback, None, 0
            )
        if not self._hook:
            log.error(
                "не удалось поставить хук клавиатуры (код %s)",
                ctypes.get_last_error(),
            )
            self._fallback_register_hotkey()
            return
        log.info("хук клавиатуры установлен (клавиша 0x%02X)", self.trigger)
        threading.Thread(target=self._warn_if_silent, daemon=True).start()
        msg = wintypes.MSG()
        while user32.GetMessageW(ctypes.byref(msg), None, 0, 0) > 0:
            user32.TranslateMessage(ctypes.byref(msg))
            user32.DispatchMessageW(ctypes.byref(msg))

    def _warn_if_silent(self) -> None:
        """Если за 40 секунд не пришло ни одной клавиши — вероятно, низкая целостность.

        Процессы, запущенные с пониженными правами (Low Mandatory Level),
        не получают клавиатурных событий вообще — Windows их не отдаёт.
        """
        time.sleep(40)
        if not self.pressed_any:
            log.warning(
                "за 40 секунд не пришло ни одного нажатия. Скорее всего программа запущена "
                "из терминала с пониженными правами (Low Integrity) — такие процессы не видят "
                "клавиатуру. Запускайте run.cmd двойным щелчком из Проводника."
            )

    def _fallback_register_hotkey(self) -> None:
        """Если хук запрещён политикой — регистрируем горячую клавишу средствами Windows.

        В этом режиме доступно только переключение (нет события отпускания),
        зато работает там, где хуки блокируются.
        """
        import ctypes
        from ctypes import wintypes

        user32 = ctypes.WinDLL("user32", use_last_error=True)
        modifiers = 0
        for mod in self.modifiers:
            modifiers |= {0x10: 0x0004, 0x11: 0x0002, 0x12: 0x0001}.get(mod, 0)
        hotkey_id = 1
        if not user32.RegisterHotKey(None, hotkey_id, modifiers, self.trigger):
            log.error(
                "RegisterHotKey тоже не сработал (код %s)", ctypes.get_last_error()
            )
            return
        log.warning(
            "хук недоступен — включён режим переключения: нажали F8 (старт), нажали ещё раз (стоп)"
        )
        first = True
        msg = wintypes.MSG()
        while user32.GetMessageW(ctypes.byref(msg), None, 0, 0) > 0:
            if msg.message == 0x0312 and msg.wParam == hotkey_id:  # WM_HOTKEY
                self._fire(self.on_press if first else self.on_release)
                first = not first
            user32.TranslateMessage(ctypes.byref(msg))
            user32.DispatchMessageW(ctypes.byref(msg))

    def _handle(self, vk: int, w_param: int) -> bool:
        down = w_param in (self.WM_KEYDOWN, self.WM_SYSKEYDOWN)
        up = w_param in (self.WM_KEYUP, self.WM_SYSKEYUP)
        if not (down or up):
            return False
        if vk in _MODIFIER_VKS:
            if down:
                self._held_modifiers.add(vk)
            else:
                self._held_modifiers.discard(vk)
            return False
        if vk == 0x1B and down:  # Esc
            self._fire(self.on_cancel)
            return False
        if vk != self.trigger:
            return False
        if down:
            self.pressed_any = True
            if self.modifiers <= self._held_modifiers:
                self._fire(self.on_press)
        else:
            self._fire(self.on_release)
        # CapsLock/NumLock/ScrollLock гасим, чтобы не переключался регистр и режимы
        return self.trigger in _SWALLOW_VKS


class App:
    def __init__(self, cfg: dict):
        self.cfg = cfg
        self.combo = [k.strip().lower() for k in str(cfg["hotkey"]).split("+")]
        self.trigger = self.combo[-1]
        self.recorder = Recorder(cfg.get("input_device"))
        self.transcriber = build_engine(cfg)
        self.jobs: queue.Queue[np.ndarray] = queue.Queue()
        self.pressed: set[str] = set()
        self.state = "idle"
        self._lock = threading.Lock()
        self._worker = threading.Thread(target=self._transcribe_loop, daemon=True)
        self._worker.start()
        self._icon = None
        self._watchdog = None
        self._last_trigger = 0.0
        self._status_text = "Готов"

    # ---------- состояние ----------
    def _set_state(self, state: str, status: str) -> None:
        self.state = state
        self._status_text = status
        if self._icon is not None:
            try:
                self._icon.icon = self._make_icon(state)
                self._icon.title = f"Vibectator: {status}"
            except Exception:  # noqa: BLE001
                pass

    @staticmethod
    def _make_icon(state: str):
        from PIL import Image, ImageDraw

        colors = {
            "idle": (90, 96, 110),
            "recording": (220, 60, 60),
            "working": (230, 170, 40),
        }

        # Основная иконка — из assets; если файла нет, рисуем запасную.
        icon_path = None
        for base in (HERE, APP_DIR):
            candidate = os.path.join(base, "assets", "icon.png")
            if os.path.exists(candidate):
                icon_path = candidate
                break
        if icon_path:
            img = Image.open(icon_path).convert("RGBA").resize((64, 64), Image.LANCZOS)
            if state != "idle":
                draw = ImageDraw.Draw(img)
                color = colors.get(state, colors["idle"])
                draw.ellipse((45, 45, 63, 63), fill=color, outline=(255, 255, 255), width=2)
            return img

        img = Image.new("RGBA", (64, 64), (0, 0, 0, 0))
        draw = ImageDraw.Draw(img)
        draw.ellipse((6, 6, 58, 58), fill=colors.get(state, colors["idle"]))
        draw.rounded_rectangle((27, 17, 37, 37), radius=5, fill=(255, 255, 255))
        draw.arc((21, 27, 43, 45), start=0, end=180, fill=(255, 255, 255), width=3)
        draw.line((32, 45, 32, 51), fill=(255, 255, 255), width=3)
        return img

    # ---------- запись ----------
    def start_recording(self) -> None:
        with self._lock:
            if self.state != "idle":
                return
            try:
                self.recorder.start()
            except Exception as exc:  # noqa: BLE001
                log.exception("микрофон недоступен")
                self._set_state("idle", "Нет доступа к микрофону")
                if self.cfg["beep"]:
                    beep("error")
                return
            self._set_state("recording", "Слушаю")
            if self.cfg["beep"]:
                beep("start")
            log.info("запись начата")
            # предохранитель: не даём записи висеть бесконечно
            limit = float(self.cfg.get("max_seconds", 60))
            self._watchdog = threading.Timer(limit, self._auto_stop)
            self._watchdog.daemon = True
            self._watchdog.start()

    def stop_recording(self, cancelled: bool = False) -> None:
        with self._lock:
            if self.state != "recording":
                return
            self._set_state("working", "Распознаю")
        self._cancel_watchdog()
        audio = self.recorder.cancel() if cancelled else self.recorder.stop()
        if cancelled:
            self._set_state("idle", "Готов")
            if self.cfg["beep"]:
                beep("cancel")
            log.info("запись отменена")
            return
        duration = len(audio) / SAMPLE_RATE
        if self.cfg["beep"]:
            beep("stop")
        if duration < float(self.cfg["min_seconds"]):
            log.info("слишком коротко: %.2fs", duration)
            self._set_state("idle", "Слишком коротко")
            return
        voiced = self._speech_seconds(audio)
        if voiced < float(self.cfg.get("min_speech_seconds", 0.3)):
            log.info("речи не найдено: %.2fs записи, %.2fs речи", duration, voiced)
            self._set_state("idle", "Речи не слышно")
            return
        log.info("запись %.2fs (речи %.2fs) -> в очередь", duration, voiced)
        self.jobs.put(audio)

    def _cancel_watchdog(self) -> None:
        if self._watchdog is not None:
            try:
                self._watchdog.cancel()
            finally:
                self._watchdog = None

    def _auto_stop(self) -> None:
        try:
            if self.state == "recording":
                log.info("достигнут предел длительности записи — останавливаю")
                self.stop_recording()
        except Exception:  # noqa: BLE001
            log.exception("ошибка в предохранителе записи")

    def _speech_seconds(self, audio: np.ndarray) -> float:
        self._set_state("working", "Проверяю речь")
        return speech_seconds(audio, self.cfg)

    # ---------- распознавание ----------
    def _transcribe_loop(self) -> None:
        while True:
            audio = self.jobs.get()
            if audio is None:
                return
            try:
                text, elapsed = self.transcriber.transcribe(audio)
                duration = len(audio) / SAMPLE_RATE
                log.info("распознано за %.2fs (аудио %.1fs, RTF %.2f): %r", elapsed, duration, elapsed / max(duration, 0.01), text)
                final = format_text(text, self.cfg)
                if final:
                    time.sleep(0.02)
                    insert_text(final, self.cfg)
                    if getattr(self, "_window_last", None) is not None:
                        try:
                            self._window_last.set(final.strip())
                        except Exception:  # noqa: BLE001
                            pass
                    if self.cfg["beep"]:
                        beep("done")
                    self._set_state("idle", "Готов")
                else:
                    self._set_state("idle", "Пусто")
            except Exception:  # noqa: BLE001
                log.exception("ошибка распознавания")
                if self.cfg["beep"]:
                    beep("error")
                self._set_state("idle", "Ошибка")
            finally:
                self.jobs.task_done()

    # ---------- хоткей ----------
    def _start_hotkey(self) -> None:
        combo = [k.strip().lower() for k in str(self.cfg["hotkey"]).split("+")]
        self.hotkey = HotkeyHook(
            combo,
            on_press=self._hotkey_press,
            on_release=self._hotkey_release,
            on_cancel=self._hotkey_cancel,
        )
        self.hotkey.start()

    def _hotkey_press(self) -> None:
        self._safe(self._on_press)

    def _hotkey_release(self) -> None:
        if self.state == "recording":
            log.info("клавиша отпущена — останавливаю запись")
            self._safe(self.stop_recording)

    def _hotkey_cancel(self) -> None:
        if self.state == "recording":
            log.info("отмена по Esc")
            self._safe(lambda: self.stop_recording(cancelled=True))

    def _safe(self, func) -> None:  # noqa: ANN001
        try:
            func()
        except Exception:  # noqa: BLE001
            log.exception("ошибка при обработке хоткея")

    def _on_press(self) -> None:
        if self.state == "idle":
            self.start_recording()
        elif self.cfg["mode"] == "toggle" and self.state == "recording":
            self.stop_recording()

    def run_window(self) -> None:
        """Простое окно с кнопкой: работает там, где перехват клавиатуры запрещён."""
        import tkinter as tk

        root = tk.Tk()
        root.title("Vibectator")
        root.geometry("380x170+80+80")
        root.attributes("-topmost", True)
        root.resizable(False, False)

        status = tk.StringVar(value="Готов")
        last = tk.StringVar(value="")
        tk.Label(root, textvariable=status, font=("Segoe UI", 11)).pack(pady=(10, 4))
        button = tk.Button(root, text="● Записать", height=2, font=("Segoe UI", 11))
        button.pack(fill="x", padx=14)
        tk.Label(root, textvariable=last, wraplength=350, fg="#555", justify="left").pack(
            pady=8, padx=10
        )

        def toggle() -> None:
            if self.state == "recording":
                self.stop_recording()
            elif self.state == "idle":
                self.start_recording()

        button.configure(command=toggle)
        self._window_button = button
        self._window_last = last

        def refresh() -> None:
            status.set(self._status_text)
            if self.state == "recording":
                button.configure(text="■ Стоп")
            elif self.state == "working":
                button.configure(text="… распознаю")
            else:
                button.configure(text="● Записать")
            root.after(150, refresh)

        root.after(150, refresh)
        # окно не должно забирать фокус: иначе текст вставится в него самого
        self._make_no_activate(root)
        log.info("окно управления открыто, хоткей не нужен")
        root.mainloop()

    def _make_no_activate(self, root) -> None:  # noqa: ANN001
        try:
            import ctypes

            root.update_idletasks()
            hwnd = ctypes.windll.user32.GetParent(root.winfo_id())
            GWL_EXSTYLE = -20
            WS_EX_NOACTIVATE = 0x08000000
            WS_EX_TOOLWINDOW = 0x00000080
            get = ctypes.windll.user32.GetWindowLongPtrW
            set_ = ctypes.windll.user32.SetWindowLongPtrW
            style = get(hwnd, GWL_EXSTYLE)
            set_(hwnd, GWL_EXSTYLE, style | WS_EX_NOACTIVATE | WS_EX_TOOLWINDOW)
        except Exception:  # noqa: BLE001
            log.debug("не удалось отключить активацию окна")

    # ---------- трей ----------
    def run_tray(self) -> None:
        self._start_hotkey()
        log.info("хоткей %s, режим %s", self.cfg["hotkey"], self.cfg["mode"])

        if self.cfg.get("tray", True):
            try:
                self._run_icon()
                return
            except Exception:  # noqa: BLE001
                log.exception("трей недоступен, работаю без иконки")
                self._icon = None
                self._notify_no_tray()
        self._run_headless()

    def _notify_no_tray(self) -> None:
        """Без иконки пользователь не видит программу — сообщаем окном."""
        try:
            import ctypes

            ctypes.windll.user32.MessageBoxW(
                None,
                "Иконка в трее недоступна, но голосовой ввод работает.\n\n"
                f"Хоткей: {self.cfg['hotkey']}\n"
                "Лог: dictation.log",
                "Vibectator",
                0x40,  # MB_ICONINFORMATION
            )
        except Exception:  # noqa: BLE001
            log.debug("не удалось показать окно-уведомление")

    def _run_icon(self) -> None:
        import pystray

        # pystray делает необязательный вызов ChangeWindowMessageFilterEx
        # (opt-in для WM_TASKBARCREATED). На части систем он падает с
        # «Access is denied» и роняет всю иконку — просто отключаем его.
        if os.name == "nt":
            try:
                from pystray._util import win32 as pystray_win32

                pystray_win32.ChangeWindowMessageFilterEx = lambda *args, **kwargs: 1
            except Exception:  # noqa: BLE001
                log.debug("не удалось отключить ChangeWindowMessageFilterEx")

        def on_toggle(icon, item):  # noqa: ANN001, ARG001
            if self.state == "idle":
                self.start_recording()
            elif self.state == "recording":
                self.stop_recording()

        def on_quit(icon, item):  # noqa: ANN001, ARG001
            icon.stop()

        menu = pystray.Menu(
            pystray.MenuItem(lambda item: f"Статус: {self._status_text}", None, enabled=False),
            pystray.MenuItem("Начать/остановить запись", on_toggle),
            pystray.MenuItem("Открыть лог", lambda icon, item: os.startfile(LOG_PATH)),  # noqa: ARG005
            pystray.Menu.SEPARATOR,
            pystray.MenuItem("Выход", on_quit),
        )
        icon = pystray.Icon("vibectator", self._make_icon("idle"), "Vibectator", menu)
        self._icon = icon
        icon.run()

    def _run_headless(self) -> None:
        """Без иконки: живём ради горячей клавиши и выходим по Ctrl+C."""
        log.info("режим без трея: работает хоткей, выход — завершить процесс")
        try:
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            log.info("остановлено пользователем")


def setup_logging() -> None:
    handler = logging.FileHandler(LOG_PATH, encoding="utf-8")
    handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
    root = logging.getLogger()
    root.setLevel(logging.INFO)
    root.addHandler(handler)
    if sys.stdout is not None and sys.stdout.isatty():
        console = logging.StreamHandler()
        console.setFormatter(logging.Formatter("%(message)s"))
        root.addHandler(console)


def transcribe_wav(path: str, cfg: dict) -> int:
    with wave.open(path, "rb") as fh:
        params = fh.getparams()
        raw = fh.readframes(params.nframes)
    if params.sampwidth != 2:
        print("нужен 16-битный WAV", file=sys.stderr)
        return 2
    audio = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
    if params.nchannels > 1:
        audio = audio.reshape(-1, params.nchannels).mean(axis=1)
    if params.framerate != SAMPLE_RATE:
        idx = np.linspace(0, len(audio) - 1, int(len(audio) * SAMPLE_RATE / params.framerate))
        audio = np.interp(idx, np.arange(len(audio)), audio).astype(np.float32)
    tr = build_engine(cfg)
    text, elapsed = tr.transcribe(audio)
    print(f"[{elapsed:.2f}s, аудио {len(audio) / SAMPLE_RATE:.1f}s] {text}")
    return 0


def _energy_speech_seconds(audio: np.ndarray) -> float:
    """Простой VAD по энергии: не требует моделей и работает в собранном виде."""
    frame = int(SAMPLE_RATE * 0.02)  # 20 мс
    if audio.size < frame:
        return 0.0
    count = audio.size // frame
    frames = audio[: count * frame].reshape(count, frame)
    rms = np.sqrt((frames ** 2).mean(axis=1))
    noise = float(np.percentile(rms, 20))
    threshold = max(noise * 3.0, 0.004)
    voiced = int((rms > threshold).sum())
    return voiced * 0.02


def speech_seconds(audio: np.ndarray, cfg: dict) -> float:
    """Сколько секунд речи в записи.

    Нужно, чтобы на тишине или фоновом шуме не появлялся выдуманный текст.
    Если доступен faster-whisper — используем Silero, иначе простой VAD по энергии.
    """
    if not cfg.get("vad", True):
        return len(audio) / SAMPLE_RATE
    try:
        from faster_whisper.vad import VadOptions, get_speech_timestamps

        segments = get_speech_timestamps(
            audio,
            VadOptions(
                threshold=0.5,
                min_speech_duration_ms=200,
                min_silence_duration_ms=300,
                speech_pad_ms=100,
            ),
            sampling_rate=SAMPLE_RATE,
        )
        return sum(seg["end"] - seg["start"] for seg in segments) / SAMPLE_RATE
    except Exception as exc:  # noqa: BLE001
        log.debug("Silero VAD недоступен (%s), считаю энергию", exc)
        return _energy_speech_seconds(audio)


def watch_keys(cfg: dict, seconds: float = 15) -> int:
    """Показывает, доходят ли нажатия до программы: удобно для диагностики."""
    combo = [k.strip().lower() for k in str(cfg["hotkey"]).split("+")]
    label = "+".join(combo).upper()
    print(f"нажмите и отпустите {label}. Наблюдаю {seconds:.0f} с...")
    done = threading.Event()

    def stamp() -> str:
        return time.strftime("%H:%M:%S")

    def on_press() -> None:
        print(f"{stamp()}  {label} нажата", flush=True)

    def on_release() -> None:
        print(f"{stamp()}  {label} отпущена", flush=True)
        done.set()

    hook = HotkeyHook(combo, on_press, on_release, on_cancel=done.set)
    hook.start()
    done.wait(seconds)
    print("готово")
    return 0 if hook.pressed_any else 1


def record_clip(cfg: dict, seconds: float) -> int:
    """Быстрая проверка: записать с микрофона и показать распознанный текст."""
    print("загружаю модель...", flush=True)
    engine = build_engine(cfg)
    recorder = Recorder(cfg.get("input_device"))
    print(f"говорите {seconds:.0f} с...", flush=True)
    if cfg["beep"]:
        beep("start")
    recorder.start()
    time.sleep(seconds)
    audio = recorder.stop()
    if cfg["beep"]:
        beep("stop")
    duration = len(audio) / SAMPLE_RATE
    voiced = speech_seconds(audio, cfg)
    if voiced < float(cfg.get("min_speech_seconds", 0.3)):
        print(f"аудио {duration:.1f}s, речи {voiced:.2f}s — речи не найдено, распознавание не запускаю")
        return 0
    text, elapsed = engine.transcribe(audio)
    print(f"аудио {duration:.1f}s (речи {voiced:.1f}s) | распознавание {elapsed:.2f}s | RTF {elapsed / max(duration, 0.01):.2f}")
    print(f"текст: {text}")
    return 0


_MUTEX_HANDLE = None


def acquire_single_instance() -> bool:
    """Второй экземпляр мешает первому: два диктофона дерутся за клавиши и микрофон."""
    global _MUTEX_HANDLE
    try:
        import ctypes

        kernel32 = ctypes.windll.kernel32
        handle = kernel32.CreateMutexW(None, False, "Local\\DSH_Dictation_SingleInstance")
        if not handle:
            return True
        if kernel32.GetLastError() == 183:  # ERROR_ALREADY_EXISTS
            return False
        _MUTEX_HANDLE = handle
        return True
    except Exception:  # noqa: BLE001
        log.debug("не удалось проверить единственный экземпляр")
        return True


def release_single_instance() -> None:
    global _MUTEX_HANDLE
    if _MUTEX_HANDLE:
        try:
            import ctypes

            ctypes.windll.kernel32.CloseHandle(_MUTEX_HANDLE)
        except Exception:  # noqa: BLE001
            pass
        _MUTEX_HANDLE = None


STOP_EVENT_NAME = "Local\\DSH_Dictation_StopEvent"


def signal_stop() -> bool:
    """Просим работающий экземпляр завершиться через именованное событие."""
    import ctypes

    kernel32 = ctypes.windll.kernel32
    handle = kernel32.OpenEventW(0x0002, False, STOP_EVENT_NAME)  # EVENT_MODIFY_STATE
    if not handle:
        return False
    try:
        return bool(kernel32.SetEvent(handle))
    finally:
        kernel32.CloseHandle(handle)


def watch_stop_event() -> None:
    import ctypes

    kernel32 = ctypes.windll.kernel32
    handle = kernel32.CreateEventW(None, False, False, STOP_EVENT_NAME)
    if not handle:
        log.warning("не удалось создать событие остановки")
        return
    while True:
        result = kernel32.WaitForSingleObject(handle, 1000)
        if result == 0:
            log.info("получен сигнал остановки — выхожу")
            os._exit(0)


def _integrity_level() -> str:
    """Уровень целостности процесса: LOW-процессы не получают события клавиатуры."""
    try:
        import ctypes
        from ctypes import wintypes

        kernel32 = ctypes.windll.kernel32
        advapi32 = ctypes.windll.advapi32
        kernel32.GetCurrentProcess.restype = wintypes.HANDLE
        advapi32.OpenProcessToken.argtypes = [
            wintypes.HANDLE,
            wintypes.DWORD,
            ctypes.POINTER(wintypes.HANDLE),
        ]
        advapi32.OpenProcessToken.restype = wintypes.BOOL
        advapi32.GetTokenInformation.argtypes = [
            wintypes.HANDLE,
            ctypes.c_int,
            ctypes.c_void_p,
            wintypes.DWORD,
            ctypes.POINTER(wintypes.DWORD),
        ]
        advapi32.GetTokenInformation.restype = wintypes.BOOL
        advapi32.GetSidSubAuthorityCount.restype = ctypes.POINTER(ctypes.c_ubyte)
        advapi32.GetSidSubAuthorityCount.argtypes = [ctypes.c_void_p]
        advapi32.GetSidSubAuthority.restype = ctypes.POINTER(ctypes.c_uint)
        advapi32.GetSidSubAuthority.argtypes = [ctypes.c_void_p, wintypes.DWORD]

        token = wintypes.HANDLE()
        if not advapi32.OpenProcessToken(
            kernel32.GetCurrentProcess(), 0x0008, ctypes.byref(token)
        ):
            return "unknown"
        size = wintypes.DWORD(0)
        advapi32.GetTokenInformation(token, 25, None, 0, ctypes.byref(size))
        if not size.value:
            return "unknown"
        buf = ctypes.create_string_buffer(size.value)
        if not advapi32.GetTokenInformation(
            token, 25, buf, size.value, ctypes.byref(size)
        ):
            return "unknown"
        sid = ctypes.cast(buf, ctypes.POINTER(ctypes.c_void_p))[0]
        count = advapi32.GetSidSubAuthorityCount(sid)[0]
        rid = advapi32.GetSidSubAuthority(sid, count - 1)[0]
        return {
            0x1000: "LOW",
            0x2000: "MEDIUM",
            0x3000: "HIGH",
            0x4000: "SYSTEM",
        }.get(rid, hex(rid))
    except Exception:  # noqa: BLE001
        return "unknown"


PID_PATH = os.path.join(HERE, "vibectator.pid")


def _pid_alive(pid: int) -> bool:
    """Жив ли процесс с таким PID (не полагаемся на списки процессов)."""
    try:
        import ctypes
        from ctypes import wintypes

        kernel32 = ctypes.windll.kernel32
        kernel32.OpenProcess.restype = wintypes.HANDLE
        handle = kernel32.OpenProcess(0x1000, False, int(pid))  # QUERY_LIMITED_INFORMATION
        if not handle:
            return False
        code = wintypes.DWORD()
        ok = kernel32.GetExitCodeProcess(handle, ctypes.byref(code))
        kernel32.CloseHandle(handle)
        return bool(ok) and code.value == 259  # STILL_ACTIVE
    except Exception:  # noqa: BLE001
        return False


def running_pid():
    try:
        with open(PID_PATH, "r", encoding="utf-8") as fh:
            pid = int(fh.read().strip())
    except Exception:  # noqa: BLE001
        return None
    return pid if _pid_alive(pid) else None


def write_pid() -> None:
    with open(PID_PATH, "w", encoding="utf-8") as fh:
        fh.write(str(os.getpid()))


def clear_pid() -> None:
    try:
        if running_pid() == os.getpid():
            os.remove(PID_PATH)
    except Exception:  # noqa: BLE001
        pass


_MUTEX_HANDLE = None


def single_instance() -> bool:
    """Вторая копия не должна запускаться: она не увидит клавиш (первая их «съедает»)."""
    global _MUTEX_HANDLE
    try:
        import ctypes

        kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
        handle = kernel32.CreateMutexW(None, False, "Local\\VibectatorSingleton")
        if not handle:
            return True
        if ctypes.get_last_error() == 183:  # ERROR_ALREADY_EXISTS
            return False
        _MUTEX_HANDLE = handle
        return True
    except Exception:  # noqa: BLE001
        return True


def main() -> int:
    parser = argparse.ArgumentParser(description="Локальный голосовой ввод (GigaAM / faster-whisper)")
    parser.add_argument("--wav", help="распознать WAV-файл и выйти")
    parser.add_argument("--record", type=float, metavar="СЕК", help="записать с микрофона и распознать")
    parser.add_argument("--insert", metavar="ТЕКСТ", help="проверить вставку готового текста в активное окно")
    parser.add_argument("--list-devices", action="store_true", help="показать микрофоны")
    parser.add_argument("--watch-keys", action="store_true", help="проверить, видит ли программа хоткей")
    parser.add_argument("--stop", action="store_true", help="остановить запущенную диктовку")
    parser.add_argument("--window", action="store_true", help="управление кнопкой в окне (без хоткея)")
    parser.add_argument("--force", action="store_true", help="игнорировать проверку единственного экземпляра")
    args = parser.parse_args()

    setup_logging()
    cfg = load_config()
    level = _integrity_level()
    log.info("уровень целостности процесса: %s", level)
    if level == "LOW":
        log.warning(
            "процесс запущен с пониженными правами (LOW): Windows не отдаёт такому "
            "процессу нажатия клавиш. Запускайте через Win+R или двойным щелчком из Проводника."
        )

    if args.stop:
        stopped = signal_stop()
        pid = running_pid()
        print(f"остановлено (PID {pid})" if stopped else "запущенной диктовки не найдено")
        if pid:
            try:
                os.remove(PID_PATH)
            except OSError:
                pass
        return 0

    if args.list_devices or args.watch_keys:
        if args.watch_keys:
            return watch_keys(cfg)
        import sounddevice as sd

        for idx, dev in enumerate(sd.query_devices()):
            if dev["max_input_channels"] > 0:
                print(f"{idx}: {dev['name']} ({dev['max_input_channels']} ch, {dev['default_samplerate']:.0f} Hz)")
        return 0

    if args.wav:
        return transcribe_wav(args.wav, cfg)

    if args.record:
        return record_clip(cfg, args.record)

    if args.insert:
        print("вставляю через 2 секунды — переключитесь в нужное окно...")
        time.sleep(2)
        insert_text(args.insert, cfg)
        print("готово")
        return 0

    other = running_pid()
    if other and not args.force:
        message = (
            f"Vibectator уже запущен (PID {other}).\n\n"
            "Если окна или иконки не видно — завершите процесс Vibectator.exe "
            "в Диспетчере задач и попробуйте снова."
        )
        log.warning("второй экземпляр отклонён: уже работает PID %s", other)
        print(message)
        if not sys.stdout or not sys.stdout.isatty():
            try:
                import ctypes

                ctypes.windll.user32.MessageBoxW(None, message, "Vibectator", 0x40)
            except Exception:  # noqa: BLE001
                pass
        return 1

    if not args.force and not single_instance():
        message = (
            "Vibectator уже работает в другой копии.\n\n"
            "Две копии мешают друг другу: одна перехватывает клавишу, вторая её уже не видит.\n"
            "Завершите Vibectator.exe в Диспетчере задач и запустите заново."
        )
        log.warning("второй экземпляр отклонён мьютексом")
        print(message)
        if not sys.stdout or not sys.stdout.isatty():
            try:
                import ctypes

                ctypes.windll.user32.MessageBoxW(None, message, "Vibectator", 0x40)
            except Exception:  # noqa: BLE001
                pass
        return 1

    app = App(cfg)
    write_pid()
    threading.Thread(target=watch_stop_event, daemon=True).start()
    try:
        if args.window:
            app.run_window()
        else:
            app.run_tray()
    finally:
        clear_pid()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
