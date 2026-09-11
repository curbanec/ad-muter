"""The two clocks in score_detector must not drift apart.

Windows are timestamped on one clock and label spans placed on another. If they
disagree, every mute is compared against labels sitting somewhere else on the
timeline, and the score is wrong in a way no assertion in the detector can
catch. The drift only appears when a chunk is not a whole number of windows --
which is every session's final chunk, and any chunk arecord cut short.
"""

from __future__ import annotations

import sys
import wave
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

from score_detector import SessionSource, session_spans, wav_duration  # noqa: E402

SAMPLE_RATE = 48000
WINDOW = 1.0


def write_wav(path: Path, seconds: float) -> Path:
    frames = int(SAMPLE_RATE * seconds)
    tone = (np.sin(np.arange(frames) / 20.0) * 8000).astype("<i2")
    with wave.open(str(path), "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(SAMPLE_RATE)
        handle.writeframes(tone.tobytes())
    return path


class NullRoku:
    timestamp = 0.0


@pytest.fixture
def two_ragged_chunks(tmp_path):
    """Two chunks of 2.5s each — deliberately not a whole number of windows."""
    pairs = []
    for n, (start, end) in enumerate(((0.0, 1.0), (1.5, 2.0))):
        wav = write_wav(tmp_path / f"chunk{n}.wav", 2.5)
        labels = tmp_path / f"chunk{n}.ads.txt"
        labels.write_text(f"{start:.6f}\t{end:.6f}\tad\n", encoding="utf-8")
        pairs.append((wav, labels))
    return pairs


def test_wav_duration_is_exact(tmp_path):
    assert wav_duration(write_wav(tmp_path / "a.wav", 2.5)) == pytest.approx(2.5)


def test_the_second_chunk_starts_where_the_first_one_ended(two_ragged_chunks):
    """A partial final window must not be credited a whole window_seconds."""
    source = SessionSource(two_ragged_chunks, WINDOW, NullRoku())
    stamps = [w.timestamp for w in source.windows()]

    # chunk 0 is 2.5s: windows at 0.0, 1.0, 2.0 (the last one half length)
    assert stamps[:3] == [0.0, 1.0, 2.0]
    # chunk 1 must begin at 2.5 — the exact end of chunk 0, not 3.0
    assert stamps[3] == pytest.approx(2.5)
    assert source.duration == pytest.approx(5.0)


def test_both_clocks_place_the_second_chunk_identically(two_ragged_chunks):
    """The window clock and the label clock must agree at every boundary."""
    source = SessionSource(two_ragged_chunks, WINDOW, NullRoku())
    stamps = [w.timestamp for w in source.windows()]
    first_of_chunk_two = stamps[3]

    spans = session_spans(two_ragged_chunks, WINDOW)
    # chunk 1's label runs 1.5-2.0 locally, so 4.0-4.5 on the session clock.
    second_span = spans[-1]
    assert second_span.start == pytest.approx(first_of_chunk_two + 1.5)
    assert second_span.start == pytest.approx(4.0)


def test_drift_accumulates_over_many_chunks(tmp_path):
    """Each ragged boundary used to add half a second; twelve chunks, six lost."""
    pairs = []
    for n in range(6):
        wav = write_wav(tmp_path / f"c{n}.wav", 2.5)
        labels = tmp_path / f"c{n}.ads.txt"
        labels.write_text("0.000000\t0.500000\tad\n", encoding="utf-8")
        pairs.append((wav, labels))

    source = SessionSource(pairs, WINDOW, NullRoku())
    list(source.windows())
    assert source.duration == pytest.approx(6 * 2.5)

    spans = session_spans(pairs, WINDOW)
    # Spans are merged when they meet within 2s, so with 0.5s of ad every 2.5s
    # what matters is that the last one sits on the exact chunk boundary.
    assert spans[-1].start == pytest.approx(5 * 2.5)
