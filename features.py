"""Мел-спектрограмма GigaAM на numpy: повторяет torchaudio.MelSpectrogram из модели.

Параметры из config модели: 16 кГц, 64 мела, win=320, hop=160, n_fft=320,
center=False, htk-шкала, нормировки нет, затем log(clamp(x, 1e-9, 1e9)).

STFT в ONNX не экспортируется, поэтому считаем признаки здесь, а в граф
отдаём готовый тензор.
"""
import numpy as np

SAMPLE_RATE = 16000
N_FFT = 320
WIN_LENGTH = 320
HOP_LENGTH = 160
N_MELS = 64


def _hann_window(n: int) -> np.ndarray:
    """Периодическое окно Ханна — как torch.hann_window(n)."""
    return 0.5 - 0.5 * np.cos(2.0 * np.pi * np.arange(n) / n)


def _hz_to_mel(freq: np.ndarray, htk: bool = True) -> np.ndarray:
    if htk:
        return 2595.0 * np.log10(1.0 + freq / 700.0)
    # slaney
    f_min, f_sp = 0.0, 200.0 / 3
    mels = (freq - f_min) / f_sp
    min_log_hz = 1000.0
    min_log_mel = (min_log_hz - f_min) / f_sp
    logstep = np.log(6.4) / 27.0
    log_part = min_log_mel + np.log(np.maximum(freq, min_log_hz) / min_log_hz) / logstep
    return np.where(freq >= min_log_hz, log_part, mels)


def _mel_to_hz(mels: np.ndarray, htk: bool = True) -> np.ndarray:
    if htk:
        return 700.0 * (10.0 ** (mels / 2595.0) - 1.0)
    f_min, f_sp = 0.0, 200.0 / 3
    freqs = f_min + f_sp * mels
    min_log_hz = 1000.0
    min_log_mel = (min_log_hz - f_min) / f_sp
    logstep = np.log(6.4) / 27.0
    log_part = min_log_hz * np.exp(logstep * (mels - min_log_mel))
    return np.where(mels >= min_log_mel, log_part, freqs)


def _mel_filterbank(sample_rate: int = SAMPLE_RATE, n_fft: int = N_FFT, n_mels: int = N_MELS) -> np.ndarray:
    """Треугольные фильтры в HTK-шкале, как torchaudio.functional.melscale_fbanks."""
    all_freqs = np.linspace(0, sample_rate // 2, n_fft // 2 + 1)
    m_min, m_max = _hz_to_mel(np.array([0.0]))[0], _hz_to_mel(np.array([sample_rate / 2.0]))[0]
    m_pts = np.linspace(m_min, m_max, n_mels + 2)
    f_pts = _mel_to_hz(m_pts)

    f_diff = f_pts[1:] - f_pts[:-1]
    slopes = f_pts[None, :] - all_freqs[:, None]
    down = -slopes[:, :-2] / f_diff[:-1]
    up = slopes[:, 2:] / f_diff[1:]
    fb = np.maximum(0.0, np.minimum(down, up))
    return fb  # (freq_bins, n_mels)


_FB = _mel_filterbank()
_WINDOW = _hann_window(WIN_LENGTH)


def log_mel(audio: np.ndarray) -> np.ndarray:
    """audio: float32 моно 16 кГц -> (1, 64, кадры) float32, как у модели."""
    audio = np.asarray(audio, dtype=np.float32).reshape(-1)
    if audio.size < WIN_LENGTH:
        audio = np.pad(audio, (0, WIN_LENGTH - audio.size))
    frames = np.lib.stride_tricks.sliding_window_view(audio, WIN_LENGTH)[::HOP_LENGTH]
    spec = np.fft.rfft(frames * _WINDOW, n=N_FFT, axis=1)
    power = spec.real ** 2 + spec.imag ** 2
    mel = power @ _FB
    mel = np.log(np.clip(mel, 1e-9, 1e9))
    return mel.T[None, :, :].astype(np.float32)


def frames_for(samples: int) -> int:
    """Сколько кадров даст сигнал такой длины (center=False)."""
    return max(1, 1 + (samples - WIN_LENGTH) // HOP_LENGTH)
