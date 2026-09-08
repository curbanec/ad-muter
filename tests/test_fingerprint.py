"""Fingerprint hashing, matching, and the authority a match carries.

Two properties matter more than the rest and are pinned hardest:

* silence must not match silence — every band energy is equal, so an unguarded
  frame hashes to 0 and any two silent stretches look identical. Ad breaks are
  full of silent seams, so unguarded this reports confident matches between
  unrelated spots. It inflated the first measured hit rate from 60% to 76%.
* streaming must equal batch — the library is built offline from whole spans
  and queried live from 1-second windows. If those two disagree by a single
  frame, nothing ever matches and the failure is completely silent.
"""

from __future__ import annotations

import numpy as np
import pytest

from admuter.config import ConfigError, DetectionConfig
from admuter.detector import Event, HeuristicDetector
from admuter.fingerprint import (
    HOP_SIZE,
    QUERY_FRAMES,
    SAMPLE_RATE,
    Ad,
    Fingerprinter,
    FingerprintIndex,
    FingerprintVoter,
    bit_error_rate,
    distinct_informative,
    is_degenerate,
)

from test_detector import AD_AFTER_GAP, CONTENT, WINDOW, warm
from test_ensemble import events


def noise(seconds: float = 12.0, seed: int = 0, gain: float = 0.1) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return (rng.standard_normal(int(SAMPLE_RATE * seconds)) * gain).astype(np.float32)


def hashes_of(audio: np.ndarray) -> list[int]:
    return Fingerprinter().feed(audio)


# --------------------------------------------------------------------------- #
# Hashing
# --------------------------------------------------------------------------- #


def test_identical_audio_hashes_identically():
    a = noise()
    assert bit_error_rate(hashes_of(a), hashes_of(a)) == 0.0


def test_unrelated_audio_sits_near_one_half():
    """Half the bits differ by chance; that is what 'no match' looks like."""
    ber = bit_error_rate(hashes_of(noise(seed=1)), hashes_of(noise(seed=2)))
    assert 0.4 < ber < 0.6


def test_a_level_change_does_not_change_the_hash():
    """The double difference cancels gain — the point of the whole scheme."""
    a = noise()
    assert bit_error_rate(hashes_of(a), hashes_of(a * 4.0)) == 0.0


def test_streaming_equals_batch():
    """The library is built in one pass and queried in 1-second slices."""
    a = noise()
    streamer = Fingerprinter()
    streamed: list[int] = []
    for i in range(0, len(a), 3000):
        streamed += streamer.feed(a[i:i + 3000])
    assert bit_error_rate(hashes_of(a), streamed) == 0.0
    assert len(streamed) == len(hashes_of(a))


def test_reset_clears_the_carry():
    f = Fingerprinter()
    f.feed(noise(2.0))
    f.reset()
    assert f.feed(noise(2.0, seed=5)) == Fingerprinter().feed(noise(2.0, seed=5))


# --------------------------------------------------------------------------- #
# Silence is not evidence
# --------------------------------------------------------------------------- #


def test_silence_hashes_to_the_degenerate_value():
    values = hashes_of(np.zeros(SAMPLE_RATE * 10, dtype=np.float32))
    assert values, "silence should still produce frames"
    assert all(is_degenerate(v) for v in values)


def test_silence_does_not_match_silence():
    """The bug that inflated the first measured hit rate from 60% to 76%."""
    silence = hashes_of(np.zeros(SAMPLE_RATE * 10, dtype=np.float32))
    index = FingerprintIndex()
    index.add(Ad(ad_id="quiet", hashes=silence))
    assert index.query(silence[:QUERY_FRAMES]) is None


def test_a_pure_tone_is_too_uniform_to_identify():
    t = np.arange(SAMPLE_RATE * 10) / SAMPLE_RATE
    tone = (0.5 * np.sin(2 * np.pi * 440 * t)).astype(np.float32)
    values = hashes_of(tone)
    assert distinct_informative(values[:QUERY_FRAMES]) < 32
    index = FingerprintIndex()
    index.add(Ad(ad_id="tone", hashes=values))
    assert index.query(values[:QUERY_FRAMES]) is None


def test_real_audio_still_matches_after_the_guard():
    values = hashes_of(noise())
    index = FingerprintIndex()
    index.add(Ad(ad_id="spot", hashes=values))
    assert index.query(values[100:100 + QUERY_FRAMES]) is not None


# --------------------------------------------------------------------------- #
# Matching
# --------------------------------------------------------------------------- #


def test_a_match_reports_where_it_is_and_what_remains():
    values = hashes_of(noise(20.0))
    index = FingerprintIndex()
    index.add(Ad(ad_id="spot", hashes=values))
    start = 200
    match = index.query(values[start:start + QUERY_FRAMES])
    assert match is not None
    assert match.ad_id == "spot"
    expected_offset = (start + QUERY_FRAMES) * HOP_SIZE / SAMPLE_RATE
    assert match.offset_seconds == pytest.approx(expected_offset, abs=0.1)
    assert match.remaining_seconds == pytest.approx(
        match.duration_seconds - expected_offset, abs=0.1
    )


def test_unrelated_audio_does_not_match():
    index = FingerprintIndex()
    index.add(Ad(ad_id="spot", hashes=hashes_of(noise(seed=1))))
    assert index.query(hashes_of(noise(seed=2))[:QUERY_FRAMES]) is None


def test_an_empty_index_matches_nothing():
    assert FingerprintIndex().query(hashes_of(noise())[:QUERY_FRAMES]) is None


def test_library_round_trips_through_disk(tmp_path):
    index = FingerprintIndex()
    index.add(Ad(ad_id="spot", hashes=hashes_of(noise()), label="test"))
    path = tmp_path / "ads.json"
    index.save(path)
    loaded = FingerprintIndex.load(path)
    assert len(loaded) == 1
    assert loaded.ads["spot"].hashes == index.ads["spot"].hashes


def test_a_library_built_at_other_framing_is_ignored(tmp_path):
    """Hashes are only comparable to audio framed identically."""
    import json

    path = tmp_path / "ads.json"
    path.write_text(json.dumps({
        "version": 1, "hop_size": HOP_SIZE * 2, "sample_rate": SAMPLE_RATE,
        "ads": [{"ad_id": "x", "hashes": [1, 2, 3]}],
    }))
    assert len(FingerprintIndex.load(path)) == 0


def test_a_missing_library_loads_empty(tmp_path):
    assert len(FingerprintIndex.load(tmp_path / "absent.json")) == 0


# --------------------------------------------------------------------------- #
# The voter
# --------------------------------------------------------------------------- #


def library_with(audio: np.ndarray, ad_id: str = "spot") -> FingerprintIndex:
    index = FingerprintIndex()
    index.add(Ad(ad_id=ad_id, hashes=hashes_of(audio)))
    return index


def test_the_voter_holds_for_the_remaining_duration():
    audio = noise(20.0)
    voter = FingerprintVoter(library_with(audio), learn=False)
    # Replay the first 8 seconds as the live service would, 1s at a time.
    for i in range(8):
        chunk = audio[i * SAMPLE_RATE:(i + 1) * SAMPLE_RATE]
        voter.feed(chunk, SAMPLE_RATE, timestamp=float(i))
    says, remaining = voter.says_ad(8.0)
    assert says is True
    assert remaining == pytest.approx(12.0, abs=1.5)
    # ...and stops of its own accord once the spot is over.
    assert voter.says_ad(21.0) == (False, 0.0)


def test_the_voter_stays_quiet_on_unknown_audio():
    voter = FingerprintVoter(library_with(noise(seed=1)), learn=False)
    unknown = noise(seed=2)
    for i in range(8):
        voter.feed(unknown[i * SAMPLE_RATE:(i + 1) * SAMPLE_RATE], SAMPLE_RATE, float(i))
    assert voter.says_ad(8.0) == (False, 0.0)


def test_resampling_from_the_capture_rate_still_matches():
    """Live audio arrives at 48 kHz stereo; the library is built at 16 kHz mono."""
    audio = noise(20.0)
    voter = FingerprintVoter(library_with(audio), learn=False)
    upsampled = np.repeat(audio, 3)  # 16k -> 48k, the inverse of the decimation
    stereo = np.stack([upsampled, upsampled], axis=1)
    for i in range(8):
        voter.feed(stereo[i * 48000:(i + 1) * 48000], 48000, float(i))
    assert voter.says_ad(8.0)[0] is True


# --------------------------------------------------------------------------- #
# Authority: a match may start a mute and set its length
# --------------------------------------------------------------------------- #


class AlwaysMatched:
    """Stands in for a voter sitting on a recognised spot with 40s to run."""

    def __init__(self, hold_until: float = 40.0) -> None:
        self._hold = hold_until

    def says_ad(self, timestamp):
        return (True, self._hold - timestamp) if timestamp < self._hold else (False, 0.0)

    @property
    def hold_until(self):
        return self._hold

    def feed(self, *_args):
        pass

    def reset(self):
        pass


def config(**overrides) -> DetectionConfig:
    base = dict(
        baseline_min_windows=5,
        fingerprint_library_path="ads.json",
        fingerprint_enabled=True,
    )
    base.update(overrides)
    return DetectionConfig(**base)


def test_a_match_starts_a_mute_without_a_transition_cue():
    """A recognised spot is its own cue: no silent seam, no warmed baseline.

    It fires on the very first window, which is the point — an identity does
    not need the baseline the inference-based voters spend 30 windows learning.
    """
    detector = HeuristicDetector(config(), WINDOW, fingerprint_voter=AlwaysMatched())
    # CONTENT-shaped audio, which the heuristic alone would never fire on.
    decision = detector.update(CONTENT, 0.0)
    assert decision.event is Event.AD_STARTED
    assert detector.in_ad


def test_a_match_carries_its_end_time_on_the_decision():
    detector = HeuristicDetector(config(), WINDOW, fingerprint_voter=AlwaysMatched())
    decision = detector.update(CONTENT, 0.0)
    assert decision.hold_until == pytest.approx(40.0)


def test_shadow_mode_records_the_match_without_acting_on_it():
    detector = HeuristicDetector(
        config(fingerprint_enabled=False, fingerprint_library_path="ads.json"),
        WINDOW, fingerprint_voter=AlwaysMatched(),
    )
    warm(detector)
    decision = detector.update(CONTENT, 8.0)
    assert decision.metrics["fingerprint_match"] == 1.0
    assert decision.event is not Event.AD_STARTED
    assert not detector.in_ad


def test_no_voter_adds_no_fingerprint_keys():
    detector = HeuristicDetector(DetectionConfig(baseline_min_windows=5), WINDOW)
    t = warm(detector)
    assert "fingerprint_match" not in detector.update(AD_AFTER_GAP, t).metrics


def test_a_match_overrides_a_vetoing_ml_voter():
    """An identity outranks an inference, in both directions."""
    class NeverAd:
        def says_ad(self, features, baseline):
            return False, 0.0

    detector = HeuristicDetector(
        config(ml_model_path="m.joblib", ml_vote_enabled=True),
        WINDOW, ml_voter=NeverAd(), fingerprint_voter=AlwaysMatched(),
    )
    assert detector.update(CONTENT, 0.0).event is Event.AD_STARTED


# --------------------------------------------------------------------------- #
# Config
# --------------------------------------------------------------------------- #


def test_enabling_without_a_library_is_a_config_error():
    with pytest.raises(ConfigError, match="no library to match against"):
        DetectionConfig(fingerprint_enabled=True).validate()


def test_an_impossible_bit_error_rate_is_rejected():
    with pytest.raises(ConfigError, match="bit error rate of unrelated audio"):
        DetectionConfig(fingerprint_max_ber=0.6).validate()


def test_a_missing_library_leaves_the_detector_working(caplog):
    from admuter.fingerprint import load_fingerprint_voter

    cfg = DetectionConfig(fingerprint_library_path="/nonexistent/ads.json")
    with caplog.at_level("WARNING"):
        voter = load_fingerprint_voter(cfg)
    assert voter is not None and len(voter.index) == 0
    assert "empty" in caplog.text


def test_no_library_path_means_no_voter():
    from admuter.fingerprint import load_fingerprint_voter

    assert load_fingerprint_voter(DetectionConfig()) is None
