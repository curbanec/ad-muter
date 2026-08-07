"""Pure, stateless audio feature extraction.

Everything here is a function of a single buffer — no history, no configuration
objects, no I/O. That keeps the module trivially testable with synthetic numpy
arrays and makes it safe to reuse from the offline replay tooling.

numpy only, by design: Phase 1 must run comfortably on a 1GB Pi 5 with no
librosa/scipy in the dependency tree.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np

# Below this we call it digital silence; log10(0) is not a useful log line.
DBFS_FLOOR = -120.0

# Anything quieter than this contributes nothing to the spectral centroid.
_SPECTRAL_EPS = 1e-12


@dataclass(frozen=True)
class Features:
    """Per-window acoustic summary.

    Levels are dBFS relative to full-scale ±1.0 float samples. ``crest_db`` is
    the peak-to-RMS ratio in dB: ~3 dB for a sine, ~0 dB for a square wave,
    12 dB+ for uncompressed dialogue. Ads are usually squashed, so a *falling*
    crest factor is one of the stronger Phase 1 cues.
    """

    rms_dbfs: float
    peak: float
    peak_dbfs: float
    crest_factor: float
    crest_db: float
    spectral_centroid_hz: float
    zero_crossing_rate: float
    is_silence: bool
    silence_ratio: float
    max_silence_run_seconds: float
    leading_silence_seconds: float
    trailing_silence_seconds: float
    interior_silence_seconds: float
    duration_seconds: float

    def as_dict(self) -> dict[str, float | bool]:
        return asdict(self)


def to_mono_float32(buffer: np.ndarray) -> np.ndarray:
    """Convert a capture buffer to mono float32 in [-1.0, 1.0].

    Accepts int16 (as delivered by ALSA/S16_LE) or float arrays, mono or
    multi-channel with shape (frames, channels).
    """
    samples = np.asarray(buffer)
    # Scale before mixing down: averaging int16 channels first would promote the
    # dtype to float64 and lose the "this was integer PCM" signal.
    if samples.dtype == np.int16:
        samples = samples.astype(np.float32) / 32768.0
    elif samples.dtype == np.int32:
        samples = samples.astype(np.float32) / 2147483648.0
    else:
        samples = samples.astype(np.float32, copy=False)
    if samples.ndim > 1:
        samples = samples.mean(axis=1)
    return np.ascontiguousarray(samples, dtype=np.float32)


def rms(samples: np.ndarray) -> float:
    """Root-mean-square amplitude, linear scale."""
    if samples.size == 0:
        return 0.0
    return float(np.sqrt(np.mean(np.square(samples, dtype=np.float64))))


def amplitude_to_dbfs(amplitude: float | np.ndarray) -> float | np.ndarray:
    """Linear amplitude -> dBFS, floored at DBFS_FLOOR instead of -inf."""
    amp = np.asarray(amplitude, dtype=np.float64)
    with np.errstate(divide="ignore"):
        db = 20.0 * np.log10(np.maximum(amp, 0.0))
    db = np.maximum(db, DBFS_FLOOR)
    db = np.nan_to_num(db, nan=DBFS_FLOOR, neginf=DBFS_FLOOR)
    if np.ndim(amplitude) == 0:
        return float(db)
    return db


def rms_dbfs(samples: np.ndarray) -> float:
    return float(amplitude_to_dbfs(rms(samples)))


def peak_level(samples: np.ndarray) -> float:
    """Absolute peak amplitude, linear scale."""
    if samples.size == 0:
        return 0.0
    return float(np.max(np.abs(samples)))


def crest_factor(peak: float, level: float) -> float:
    """Peak/RMS ratio. Degenerate all-zero buffers report 1.0 (0 dB)."""
    if level <= 0.0:
        return 1.0
    return float(peak / level)


def spectral_centroid(samples: np.ndarray, sample_rate: int) -> float:
    """Magnitude-weighted mean frequency in Hz (0.0 for a silent buffer)."""
    if samples.size == 0:
        return 0.0
    window = np.hanning(samples.size).astype(np.float32)
    spectrum = np.abs(np.fft.rfft(samples * window))
    total = float(spectrum.sum())
    if total <= _SPECTRAL_EPS:
        return 0.0
    freqs = np.fft.rfftfreq(samples.size, d=1.0 / sample_rate)
    return float(np.dot(freqs, spectrum) / total)


def zero_crossing_rate(samples: np.ndarray) -> float:
    """Fraction of adjacent sample pairs that change sign."""
    if samples.size < 2:
        return 0.0
    signs = np.signbit(samples)
    return float(np.count_nonzero(signs[1:] != signs[:-1]) / (samples.size - 1))


def frame_rms(samples: np.ndarray, frame_length: int) -> np.ndarray:
    """RMS of each complete frame; the whole buffer as one frame if it is short."""
    if samples.size == 0:
        return np.zeros(0, dtype=np.float64)
    if frame_length < 1:
        frame_length = 1
    n_frames = samples.size // frame_length
    if n_frames == 0:
        return np.array([rms(samples)], dtype=np.float64)
    trimmed = samples[: n_frames * frame_length].reshape(n_frames, frame_length)
    return np.sqrt(np.mean(np.square(trimmed, dtype=np.float64), axis=1))


def _runs(mask: np.ndarray) -> list[tuple[int, int]]:
    """Start/stop index pairs (half-open) of every True run in a bool mask."""
    if mask.size == 0:
        return []
    padded = np.concatenate(([False], mask, [False]))
    edges = np.flatnonzero(padded[1:] != padded[:-1])
    return list(zip(edges[0::2].tolist(), edges[1::2].tolist()))


def silence_profile(
    silent_frames: np.ndarray, frame_seconds: float
) -> tuple[float, float, float, float, float]:
    """Summarize where the silence sits inside a window.

    Returns ``(ratio, max_run, leading, trailing, interior)`` in seconds (ratio
    is dimensionless). ``interior`` is the longest silent run that touches
    neither edge — i.e. a gap we know is complete, without needing to look at
    the neighbouring windows.

    Sub-window resolution matters: a 0.3 s ad gap never makes a whole 1 s window
    read as silent, so a window-level flag alone would miss most transitions.
    """
    n = int(silent_frames.size)
    if n == 0:
        return 0.0, 0.0, 0.0, 0.0, 0.0
    ratio = float(np.count_nonzero(silent_frames) / n)
    runs = _runs(silent_frames)
    if not runs:
        return 0.0, 0.0, 0.0, 0.0, 0.0
    max_run = max(stop - start for start, stop in runs) * frame_seconds
    leading = (runs[0][1] - runs[0][0]) * frame_seconds if runs[0][0] == 0 else 0.0
    trailing = (runs[-1][1] - runs[-1][0]) * frame_seconds if runs[-1][1] == n else 0.0
    interior_runs = [
        (stop - start) for start, stop in runs if start > 0 and stop < n
    ]
    interior = max(interior_runs) * frame_seconds if interior_runs else 0.0
    return ratio, float(max_run), float(leading), float(trailing), float(interior)


def compute_features(
    samples: np.ndarray,
    sample_rate: int,
    silence_dbfs: float = -60.0,
    frame_seconds: float = 0.05,
) -> Features:
    """Extract every Phase 1 feature from one mono float32 window."""
    mono = to_mono_float32(samples)
    duration = float(mono.size / sample_rate) if sample_rate > 0 else 0.0

    level = rms(mono)
    level_db = float(amplitude_to_dbfs(level))
    peak = peak_level(mono)
    crest = crest_factor(peak, level)

    frame_length = max(1, int(round(sample_rate * frame_seconds)))
    frames = frame_rms(mono, frame_length)
    frame_db = np.asarray(amplitude_to_dbfs(frames), dtype=np.float64)
    silent_frames = frame_db < silence_dbfs
    effective_frame_seconds = (
        frame_length / sample_rate if frames.size > 1 else duration
    )
    ratio, max_run, leading, trailing, interior = silence_profile(
        silent_frames, effective_frame_seconds
    )

    return Features(
        rms_dbfs=level_db,
        peak=peak,
        peak_dbfs=float(amplitude_to_dbfs(peak)),
        crest_factor=crest,
        crest_db=float(amplitude_to_dbfs(crest)),
        spectral_centroid_hz=spectral_centroid(mono, sample_rate),
        zero_crossing_rate=zero_crossing_rate(mono),
        is_silence=bool(level_db < silence_dbfs),
        silence_ratio=ratio,
        max_silence_run_seconds=max_run,
        leading_silence_seconds=leading,
        trailing_silence_seconds=trailing,
        interior_silence_seconds=interior,
        duration_seconds=duration,
    )
