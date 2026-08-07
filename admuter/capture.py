"""Blocking ALSA capture that yields fixed-size analysis windows.

``sounddevice`` is imported lazily so that the rest of the package — features,
detector, replay tooling, the whole test suite — works on machines with no
PortAudio and no sound card.

The TV being switched off is a normal condition, not an error: the device can
vanish mid-stream and reappear later. We log, back off, and reconnect. The first
window after any (re)connection is flagged so the controller can reset state
rather than compare against a baseline learned before the gap.
"""

from __future__ import annotations

import logging
import re
import time
import wave
from collections.abc import Callable, Iterator
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .config import AudioConfig
from .features import to_mono_float32

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class AudioWindow:
    """One analysis window of mono float32 audio."""

    samples: np.ndarray
    sample_rate: int
    index: int
    timestamp: float
    stream_restarted: bool = False


class CaptureError(RuntimeError):
    """Raised only for unrecoverable setup problems (never mid-loop)."""


def _import_sounddevice():  # pragma: no cover - trivial, needs PortAudio
    try:
        import sounddevice as sd
    except (ImportError, OSError) as exc:
        raise CaptureError(
            "sounddevice/PortAudio is unavailable. On Raspberry Pi OS: "
            "sudo apt install libportaudio2 && pip install sounddevice"
        ) from exc
    return sd


def resolve_device(spec: str, query_devices: Callable[..., object] | None = None):
    """Turn a config device string into something PortAudio accepts.

    ALSA names like ``plughw:CARD=Receiver,DEV=0`` are not PortAudio device
    names, so we also try the bare card name (``Receiver``), which sounddevice
    resolves by case-insensitive substring match. A plain integer is used as a
    device index.
    """
    if query_devices is None:  # pragma: no cover - needs PortAudio
        query_devices = _import_sounddevice().query_devices

    text = spec.strip()
    if text.isdigit():
        return int(text)

    candidates = [text]
    card = re.search(r"CARD=([^,]+)", text)
    if card:
        candidates.append(card.group(1))

    errors: list[str] = []
    for candidate in candidates:
        try:
            info = query_devices(candidate, "input")
        except Exception as exc:  # sounddevice raises ValueError for no match
            errors.append(f"{candidate!r}: {exc}")
            continue
        name = info.get("name", candidate) if isinstance(info, dict) else candidate
        log.info("audio device %r resolved to %r", spec, name)
        return candidate
    raise CaptureError(
        f"no input device matched {spec!r} ({'; '.join(errors)}). "
        "Run `python -m sounddevice` to list what PortAudio can see."
    )


class AudioCapture:
    """Yields ~window_seconds buffers from the configured input device."""

    def __init__(
        self,
        config: AudioConfig,
        clock: Callable[[], float] = time.monotonic,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        self.config = config
        self._clock = clock
        self._sleep = sleep
        self._stop = False
        self._index = 0

    def stop(self) -> None:
        """Ask the generator to finish after the current read."""
        self._stop = True

    def windows(self) -> Iterator[AudioWindow]:
        """Yield analysis windows forever, reconnecting as needed."""
        sd = _import_sounddevice()
        cfg = self.config
        frames = cfg.frames_per_window
        backoff = cfg.retry_initial_seconds
        restarted = True

        while not self._stop:
            try:
                device = resolve_device(cfg.device, sd.query_devices)
                stream = sd.InputStream(
                    device=device,
                    channels=cfg.channels,
                    samplerate=cfg.sample_rate,
                    dtype="int16",
                    blocksize=frames,
                    latency="high",
                )
                with stream:
                    log.info(
                        "capturing from %s at %d Hz, %d ch, %.2fs windows",
                        cfg.device,
                        cfg.sample_rate,
                        cfg.channels,
                        cfg.window_seconds,
                    )
                    backoff = cfg.retry_initial_seconds
                    while not self._stop:
                        data, overflowed = stream.read(frames)
                        if overflowed:
                            log.warning("audio input overflow — dropped samples")
                        window = AudioWindow(
                            samples=to_mono_float32(data),
                            sample_rate=cfg.sample_rate,
                            index=self._index,
                            timestamp=self._clock(),
                            stream_restarted=restarted,
                        )
                        self._index += 1
                        restarted = False
                        yield window
            except GeneratorExit:  # pragma: no cover - consumer went away
                raise
            except (CaptureError, sd.PortAudioError, OSError, ValueError) as exc:
                if self._stop:
                    break
                log.warning(
                    "audio capture unavailable (%s); retrying in %.1fs "
                    "(is the TV on?)",
                    exc,
                    backoff,
                )
                self._sleep(backoff)
                backoff = min(backoff * 2.0, cfg.retry_max_seconds)
                restarted = True
        log.info("audio capture stopped after %d windows", self._index)


def wav_windows(
    path: str | Path,
    window_seconds: float = 1.0,
    start_timestamp: float = 0.0,
) -> Iterator[AudioWindow]:
    """Iterate a WAV file as AudioWindows — the offline twin of AudioCapture.

    Used by ``scripts/replay_wav.py`` and by tests, so threshold tuning runs on
    exactly the same feature path as the live service.
    """
    path = Path(path)
    with wave.open(str(path), "rb") as wav:
        sample_rate = wav.getframerate()
        channels = wav.getnchannels()
        width = wav.getsampwidth()
        if width != 2:
            raise CaptureError(
                f"{path}: expected 16-bit PCM, got {width * 8}-bit. "
                "Record with -f S16_LE."
            )
        frames_per_window = max(1, int(round(sample_rate * window_seconds)))
        index = 0
        while True:
            raw = wav.readframes(frames_per_window)
            if not raw:
                break
            data = np.frombuffer(raw, dtype="<i2")
            if channels > 1:
                usable = (data.size // channels) * channels
                data = data[:usable].reshape(-1, channels)
            if data.size == 0:
                break
            yield AudioWindow(
                samples=to_mono_float32(data),
                sample_rate=sample_rate,
                index=index,
                timestamp=start_timestamp + index * window_seconds,
                stream_restarted=index == 0,
            )
            index += 1
