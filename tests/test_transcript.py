"""Transcript voter: lexicon, decay, resampling, and the stay-only rule.

Vosk publishes no macOS wheel, so the real recogniser can only be exercised on
the Pi. Everything around it — decimation, the bounded hand-off, decay, scoring
and the wiring into the detector — sits behind a Protocol and is tested here.
"""

from __future__ import annotations

import numpy as np
import pytest

from admuter.config import ConfigError, DetectionConfig
from admuter.detector import Event, HeuristicDetector
from admuter.transcript import (
    ASR_SAMPLE_RATE,
    LexiconScorer,
    TranscriptVoter,
    normalise,
    to_asr_pcm,
)

from test_detector import AD, AD_AFTER_GAP, CONTENT, WINDOW, warm
from test_ensemble import events


class FakeRecognizer:
    """Returns queued transcripts, one per accepted chunk."""

    def __init__(self, *transcripts: str) -> None:
        self.pending = list(transcripts)
        self.accepted = 0

    def accept(self, pcm16: bytes) -> str:
        self.accepted += 1
        return self.pending.pop(0) if self.pending else ""

    def final(self) -> str:
        return ""


# --------------------------------------------------------------------------- #
# Lexicon
# --------------------------------------------------------------------------- #


def test_normalise_flattens_asr_output():
    assert normalise("Ask your DOCTOR, today!") == "ask your doctor today"


def test_two_distinct_phrases_score():
    hit = LexiconScorer().score(
        "ask your doctor about it side effects may include headache"
    )
    assert hit.score > 0
    assert len(hit.phrases) >= 2


def test_a_single_phrase_is_not_enough():
    """A medical drama really does say 'tell your doctor'."""
    hit = LexiconScorer().score(
        "you need to tell your doctor what happened to you last night"
    )
    assert hit.score == 0.0
    assert hit.phrases == ("tell your doctor",)


def test_ordinary_dialogue_scores_nothing():
    for line in [
        "i cannot believe you did that to me at the wedding",
        "we are going to need a bigger table for thanksgiving",
        "the kids are asleep so keep your voice down please",
    ]:
        assert LexiconScorer().score(line).score == 0.0


def test_real_shaped_ad_copy_scores_well():
    hit = LexiconScorer().score(
        "ask your doctor about this prescription treatment serious side effects "
        "may include nausea results may vary"
    )
    assert hit.score >= 3.0


def test_min_phrases_is_configurable():
    text = "call now for a free quote"
    assert LexiconScorer(min_phrases=3).score(text).score == 0.0
    assert LexiconScorer(min_phrases=2).score(text).score > 0.0


# --------------------------------------------------------------------------- #
# Resampling
# --------------------------------------------------------------------------- #


def test_decimates_48k_stereo_to_16k_mono_int16():
    samples = np.zeros((4800, 2), dtype=np.float32)
    pcm = to_asr_pcm(samples, 48000)
    assert len(pcm) == (4800 // 3) * 2  # int16 mono at a third the rate
    assert np.frombuffer(pcm, dtype="<i2").max() == 0


def test_already_16k_mono_passes_through():
    samples = np.zeros(1600, dtype=np.float32)
    assert len(to_asr_pcm(samples, ASR_SAMPLE_RATE)) == 1600 * 2


def test_clipping_does_not_wrap_around():
    """A sample above 1.0 must saturate, not overflow into a loud negative."""
    pcm = np.frombuffer(to_asr_pcm(np.full(3, 4.0, dtype=np.float32), 16000), "<i2")
    assert (pcm > 0).all()


def test_a_window_shorter_than_the_factor_yields_nothing():
    assert to_asr_pcm(np.zeros(2, dtype=np.float32), 48000) == b""


# --------------------------------------------------------------------------- #
# Decay
# --------------------------------------------------------------------------- #


def voter_with(text: str, **kwargs) -> TranscriptVoter:
    """Run one chunk through synchronously, without starting the thread."""
    voter = TranscriptVoter(FakeRecognizer(text), **kwargs)
    voter.feed(np.zeros(4800, dtype=np.float32), 48000, timestamp=100.0)
    voter._queue.put_nowait(None)
    voter._run()
    return voter


def test_a_hit_votes_yes_immediately():
    voter = voter_with("ask your doctor side effects may include", threshold=2.0)
    assert voter.says_ad(100.0)[0] is True


def test_the_vote_decays_to_nothing():
    voter = voter_with(
        "ask your doctor side effects may include", threshold=2.0, decay_seconds=30.0
    )
    fresh = voter.says_ad(100.0)[1]
    later = voter.says_ad(115.0)[1]
    assert 0 < later < fresh
    assert voter.says_ad(130.0) == (False, 0.0)
    assert voter.says_ad(200.0) == (False, 0.0)


def test_evidence_from_the_future_is_ignored():
    """Clock going backwards (a chunk boundary, a restart) must not vote."""
    voter = voter_with("ask your doctor side effects may include", threshold=2.0)
    assert voter.says_ad(50.0) == (False, 0.0)


def test_dialogue_never_votes():
    voter = voter_with("i cannot believe you said that to my mother")
    assert voter.says_ad(100.0) == (False, 0.0)


# --------------------------------------------------------------------------- #
# The audio path must not block
# --------------------------------------------------------------------------- #


def test_feed_drops_rather_than_blocking_when_the_recogniser_falls_behind():
    voter = TranscriptVoter(FakeRecognizer(), queue_size=2)
    chunk = np.zeros(4800, dtype=np.float32)
    for i in range(10):  # nothing is consuming; the queue fills at 2
        voter.feed(chunk, 48000, timestamp=float(i))
    assert voter.stats["asr_dropped_windows"] == 8


def test_a_broken_recogniser_does_not_kill_the_worker():
    class Exploding:
        def accept(self, pcm16):
            raise RuntimeError("boom")

        def final(self):
            return ""

    voter = TranscriptVoter(Exploding())
    voter.feed(np.zeros(4800, dtype=np.float32), 48000, timestamp=1.0)
    voter._queue.put_nowait(None)
    voter._run()  # must return, not raise
    assert voter.says_ad(1.0) == (False, 0.0)


# --------------------------------------------------------------------------- #
# Stay-only wiring
# --------------------------------------------------------------------------- #


class AlwaysAd:
    def says_ad(self, timestamp):
        return True, 99.0


def test_the_transcript_cannot_start_a_mute():
    """It lags by 10-20s; entering on it would mute the show after the break."""
    config = DetectionConfig(
        baseline_min_windows=5, asr_model_path="fake", asr_vote_enabled=True
    )
    detector = HeuristicDetector(config, WINDOW, transcript_voter=AlwaysAd())
    assert Event.AD_STARTED not in events(detector, CONTENT, 30, 0.0)
    assert not detector.in_ad


def test_the_transcript_extends_a_mute_the_heuristic_would_have_dropped():
    config = dict(baseline_min_windows=5, ad_end_windows=2, min_ad_seconds=0.0)

    plain = HeuristicDetector(DetectionConfig(**config), WINDOW)
    t = warm(plain)
    assert plain.update(AD_AFTER_GAP, t).event is Event.AD_STARTED
    assert Event.AD_ENDED in events(plain, CONTENT, 4, t + WINDOW)

    held = HeuristicDetector(
        DetectionConfig(asr_model_path="fake", asr_vote_enabled=True, **config),
        WINDOW,
        transcript_voter=AlwaysAd(),
    )
    t = warm(held)
    assert held.update(AD_AFTER_GAP, t).event is Event.AD_STARTED
    assert Event.AD_ENDED not in events(held, CONTENT, 4, t + WINDOW)
    assert held.in_ad


def test_shadow_mode_records_but_does_not_extend():
    config = dict(baseline_min_windows=5, ad_end_windows=2, min_ad_seconds=0.0)
    shadowed = HeuristicDetector(
        DetectionConfig(asr_model_path="fake", asr_vote_enabled=False, **config),
        WINDOW,
        transcript_voter=AlwaysAd(),
    )
    t = warm(shadowed)
    assert shadowed.update(AD_AFTER_GAP, t).event is Event.AD_STARTED
    assert Event.AD_ENDED in events(shadowed, CONTENT, 4, t + WINDOW)
    assert not shadowed.in_ad

    last = shadowed.update(CONTENT, t + 9 * WINDOW)
    assert last.metrics["asr_says_ad"] == 1.0
    assert last.metrics["asr_score"] == pytest.approx(99.0)


def test_no_voter_adds_no_asr_keys():
    detector = HeuristicDetector(DetectionConfig(baseline_min_windows=5), WINDOW)
    t = warm(detector)
    metrics = detector.update(AD_AFTER_GAP, t).metrics
    assert "asr_score" not in metrics and "asr_says_ad" not in metrics


# --------------------------------------------------------------------------- #
# Config
# --------------------------------------------------------------------------- #


def test_enabling_asr_without_a_model_is_a_config_error():
    with pytest.raises(ConfigError, match="no recogniser to vote with"):
        DetectionConfig(asr_vote_enabled=True).validate()


def test_a_missing_asr_model_leaves_the_detector_working(caplog):
    from admuter.transcript import load_transcript_voter

    config = DetectionConfig(asr_model_path="/nonexistent/vosk-model")
    with caplog.at_level("WARNING"):
        assert load_transcript_voter(config) is None
    assert "ASR voter unavailable" in caplog.text


def test_no_asr_path_starts_no_thread():
    from admuter.transcript import load_transcript_voter

    assert load_transcript_voter(DetectionConfig()) is None


def test_synchronous_mode_recognises_inline_without_a_thread():
    """Offline replay needs determinism, not a thread that drops what it misses."""
    voter = TranscriptVoter(
        FakeRecognizer("ask your doctor side effects may include"),
        threshold=2.0,
        synchronous=True,
    )
    voter.feed(np.zeros(4800, dtype=np.float32), 48000, timestamp=100.0)
    assert voter._thread is None
    assert voter.says_ad(100.0)[0] is True
    assert voter.stats["asr_dropped_windows"] == 0
