"""Detector tests — synthetic Features, no audio, no hardware."""

from __future__ import annotations

import dataclasses

import pytest

from admuter.config import Config, ConfigError, DetectionConfig
from admuter.detector import Event, HeuristicDetector
from admuter.features import Features

WINDOW = 1.0


def make_features(
    rms_dbfs: float = -20.0,
    crest_db: float = 12.0,
    centroid_hz: float = 2000.0,
    is_silence: bool = False,
    silence_ratio: float = 0.0,
    leading: float = 0.0,
    trailing: float = 0.0,
    interior: float = 0.0,
    duration: float = WINDOW,
) -> Features:
    return Features(
        rms_dbfs=rms_dbfs,
        peak=0.5,
        peak_dbfs=rms_dbfs + crest_db,
        crest_factor=10 ** (crest_db / 20.0),
        crest_db=crest_db,
        spectral_centroid_hz=centroid_hz,
        zero_crossing_rate=0.1,
        is_silence=is_silence,
        silence_ratio=silence_ratio,
        max_silence_run_seconds=max(leading, trailing, interior),
        leading_silence_seconds=leading,
        trailing_silence_seconds=trailing,
        interior_silence_seconds=interior,
        duration_seconds=duration,
    )


CONTENT = make_features()
SILENT_WINDOW = make_features(
    rms_dbfs=-90.0,
    crest_db=0.0,
    is_silence=True,
    silence_ratio=1.0,
    leading=WINDOW,
    trailing=WINDOW,
)
# Louder than baseline and squashed: the ad profile.
AD = make_features(rms_dbfs=-15.0, crest_db=7.0, centroid_hz=2600.0)
# Louder and squashed, arriving after a 0.4s gap inside the window.
AD_AFTER_GAP = dataclasses.replace(AD, interior_silence_seconds=0.4, silence_ratio=0.4)


@pytest.fixture
def config() -> DetectionConfig:
    return DetectionConfig(baseline_min_windows=5)


@pytest.fixture
def detector(config: DetectionConfig) -> HeuristicDetector:
    return HeuristicDetector(config, WINDOW)


def feed(detector: HeuristicDetector, features: Features, count: int, start: float = 0.0):
    """Feed `count` copies of one window; return the last decision and next time."""
    decision = None
    t = start
    for _ in range(count):
        decision = detector.update(features, t)
        t += WINDOW
    return decision, t


def warm(detector: HeuristicDetector, windows: int = 8) -> float:
    """Establish the content baseline; returns the next timestamp."""
    _, t = feed(detector, CONTENT, windows)
    return t


# --------------------------------------------------------------------------- #
# Baseline
# --------------------------------------------------------------------------- #


def test_no_events_before_the_baseline_is_ready(detector):
    for i in range(4):
        decision = detector.update(AD_AFTER_GAP, i * WINDOW)
        assert decision.event is Event.NO_CHANGE
        assert "baseline warming up" in decision.reason


def test_baseline_tracks_content(detector):
    warm(detector)
    assert detector.baseline.ready(5)
    assert detector.baseline.rms_dbfs == pytest.approx(-20.0)
    assert detector.baseline.crest_db == pytest.approx(12.0)


def test_silence_does_not_poison_the_baseline(detector):
    warm(detector)
    feed(detector, SILENT_WINDOW, 3, start=100.0)
    assert detector.baseline.rms_dbfs == pytest.approx(-20.0)


def test_steady_content_never_fires(detector):
    decision, _ = feed(detector, CONTENT, 60)
    assert decision.event is Event.NO_CHANGE
    assert decision.ad_profile is False
    assert not detector.in_ad


# --------------------------------------------------------------------------- #
# Ad start
# --------------------------------------------------------------------------- #


def test_gap_plus_ad_profile_starts_an_ad(detector):
    t = warm(detector)
    decision = detector.update(AD_AFTER_GAP, t)
    assert decision.event is Event.AD_STARTED
    assert decision.ad_profile is True
    assert decision.confidence > 0.0
    assert decision.metrics["gap_seconds"] == pytest.approx(0.4)
    assert detector.in_ad


def test_gap_split_across_two_windows_still_counts(detector):
    """A silent stretch spanning a window boundary is stitched, not dropped."""
    t = warm(detector)
    decision = detector.update(SILENT_WINDOW, t)
    assert decision.event is Event.NO_CHANGE
    ad_with_leading_gap = dataclasses.replace(
        AD, leading_silence_seconds=0.3, silence_ratio=0.3
    )
    decision = detector.update(ad_with_leading_gap, t + WINDOW)
    assert decision.event is Event.AD_STARTED
    assert decision.metrics["gap_seconds"] == pytest.approx(1.3)


def test_ad_profile_without_a_transition_cue_does_not_fire(detector):
    t = warm(detector)
    decision = detector.update(AD, t)
    assert decision.event is Event.NO_CHANGE
    assert decision.ad_profile is True  # loud and squashed...
    assert "without a transition cue" in decision.reason  # ...but no seam
    assert not detector.in_ad


def test_gap_without_an_abrupt_shift_does_not_arm_the_cue(detector):
    """A quiet beat in dialogue is a gap, but nothing changes across it."""
    t = warm(detector)
    same_after_gap = dataclasses.replace(
        CONTENT, interior_silence_seconds=0.5, silence_ratio=0.5
    )
    decision = detector.update(same_after_gap, t)
    assert decision.metrics["cue_active"] == 0.0
    decision = detector.update(AD, t + WINDOW)
    assert decision.event is Event.NO_CHANGE


def test_gap_longer_than_max_is_not_a_transition_cue(detector):
    t = warm(detector)
    feed(detector, SILENT_WINDOW, 3, start=t)  # 3s of silence: a pause, not a seam
    decision = detector.update(
        dataclasses.replace(AD, leading_silence_seconds=0.5, silence_ratio=0.5),
        t + 3 * WINDOW,
    )
    assert decision.event is Event.NO_CHANGE


def test_cue_expires_after_the_grace_period(detector):
    cfg = DetectionConfig(baseline_min_windows=5, cue_grace_seconds=2.0)
    detector = HeuristicDetector(cfg, WINDOW)
    t = warm(detector)
    # A seam, but the audio that follows only looks ad-like much later.
    seam = dataclasses.replace(
        CONTENT, interior_silence_seconds=0.5, silence_ratio=0.5, rms_dbfs=-30.0
    )
    detector.update(seam, t)
    decision = detector.update(AD, t + 5 * WINDOW)
    assert decision.event is Event.NO_CHANGE
    assert not detector.in_ad


def test_require_transition_cue_false_fires_on_profile_alone(config):
    detector = HeuristicDetector(
        dataclasses.replace(config, require_transition_cue=False), WINDOW
    )
    t = warm(detector)
    assert detector.update(AD, t).event is Event.AD_STARTED


def test_louder_but_still_dynamic_audio_is_not_an_ad(detector):
    """A loud action scene keeps its dynamics — crest stays high, so no mute."""
    t = warm(detector)
    loud_scene = make_features(
        rms_dbfs=-12.0, crest_db=13.0, interior=0.4, silence_ratio=0.4
    )
    decision = detector.update(loud_scene, t)
    assert decision.event is Event.NO_CHANGE
    assert decision.ad_profile is False


# --------------------------------------------------------------------------- #
# Ad end
# --------------------------------------------------------------------------- #


def test_ad_ends_when_the_profile_returns_to_content(detector):
    t = warm(detector)
    detector.update(AD_AFTER_GAP, t)
    t += WINDOW
    decision, t = feed(detector, AD, 8, start=t)
    assert decision.event is Event.NO_CHANGE
    assert detector.in_ad

    detector.update(CONTENT, t)  # first non-ad window
    decision = detector.update(CONTENT, t + WINDOW)  # second: ad_end_windows=2
    assert decision.event is Event.AD_ENDED
    assert "non-ad windows" in decision.reason
    assert not detector.in_ad


def test_min_ad_seconds_prevents_instant_flapping(detector):
    t = warm(detector)
    detector.update(AD_AFTER_GAP, t)
    # min_ad_seconds defaults to 5s, so two quiet windows at t+1/t+2 must not end it.
    detector.update(CONTENT, t + WINDOW)
    decision = detector.update(CONTENT, t + 2 * WINDOW)
    assert decision.event is Event.NO_CHANGE
    assert detector.in_ad


def test_failsafe_ends_the_ad_after_max_ad_seconds(config):
    detector = HeuristicDetector(
        dataclasses.replace(config, max_ad_seconds=10.0), WINDOW
    )
    t = warm(detector)
    detector.update(AD_AFTER_GAP, t)
    decision, _ = feed(detector, AD, 9, start=t + WINDOW)
    assert decision.event is Event.NO_CHANGE
    decision = detector.update(AD, t + 10 * WINDOW)
    assert decision.event is Event.AD_ENDED
    assert "failsafe" in decision.reason
    assert not detector.in_ad


def test_baseline_reseeds_after_an_ad_ends(detector):
    t = warm(detector)
    detector.update(AD_AFTER_GAP, t)
    _, t = feed(detector, AD, 8, start=t + WINDOW)
    quieter_content = make_features(rms_dbfs=-24.0, crest_db=14.0)
    detector.update(quieter_content, t)
    before = detector.baseline.rms_dbfs
    detector.update(quieter_content, t + WINDOW)  # AD_ENDED window re-seeds
    assert detector.baseline.rms_dbfs < before


# --------------------------------------------------------------------------- #
# Hysteresis
# --------------------------------------------------------------------------- #

# +2 dB over the -20 dBFS content baseline, squashed like an ad. Under the
# hysteresis config below that clears the 1 dB "stay" bar but not the 3 dB
# "enter" bar.
MARGINAL = make_features(rms_dbfs=-18.0, crest_db=7.0, centroid_hz=2600.0)
MARGINAL_AFTER_GAP = dataclasses.replace(
    MARGINAL, interior_silence_seconds=0.4, silence_ratio=0.4
)


@pytest.fixture
def hysteresis_detector() -> HeuristicDetector:
    cfg = DetectionConfig(
        baseline_min_windows=5, ad_loudness_delta_db=3.0, ad_stay_loudness_delta_db=1.0
    )
    return HeuristicDetector(cfg, WINDOW)


def test_marginal_loudness_does_not_enter_an_ad(hysteresis_detector):
    t = warm(hysteresis_detector)
    decision = hysteresis_detector.update(MARGINAL_AFTER_GAP, t)
    assert decision.event is Event.NO_CHANGE
    assert decision.ad_profile is False
    assert decision.metrics["loudness_delta_db"] == pytest.approx(2.0)
    assert decision.metrics["ad_loudness_delta_db"] == pytest.approx(3.0)
    assert decision.metrics["ad_stay_loudness_delta_db"] == pytest.approx(1.0)
    assert not hysteresis_detector.in_ad


def test_marginal_loudness_keeps_an_ad_going_once_entered(hysteresis_detector):
    t = warm(hysteresis_detector)
    assert hysteresis_detector.update(AD_AFTER_GAP, t).event is Event.AD_STARTED
    # Well past min_ad_seconds and ad_end_windows: only the stay bar holds it open.
    decision, _ = feed(hysteresis_detector, MARGINAL, 8, start=t + WINDOW)
    assert decision.event is Event.NO_CHANGE
    assert decision.ad_profile is True
    assert "ad continues" in decision.reason
    assert hysteresis_detector.in_ad


def test_without_hysteresis_the_same_windows_end_the_ad():
    """Control: stay == enter reproduces the old single-threshold behaviour."""
    cfg = DetectionConfig(
        baseline_min_windows=5, ad_loudness_delta_db=3.0, ad_stay_loudness_delta_db=3.0
    )
    detector = HeuristicDetector(cfg, WINDOW)
    t = warm(detector)
    detector.update(AD_AFTER_GAP, t)
    _, t = feed(detector, AD, 5, start=t + WINDOW)  # clear min_ad_seconds
    decision, _ = feed(detector, MARGINAL, 2, start=t)  # ad_end_windows=2
    assert decision.event is Event.AD_ENDED
    assert not detector.in_ad


def test_stay_threshold_defaults_to_two_db_under_enter():
    detection = Config.from_dict({"detection": {"ad_loudness_delta_db": 3.5}}).detection
    assert detection.ad_stay_loudness_delta_db == pytest.approx(1.5)


def test_stay_threshold_above_enter_threshold_is_rejected():
    with pytest.raises(ConfigError, match="ad_stay_loudness_delta_db"):
        Config.from_dict(
            {"detection": {"ad_loudness_delta_db": 2.0, "ad_stay_loudness_delta_db": 2.5}}
        )


# --------------------------------------------------------------------------- #
# Lifecycle
# --------------------------------------------------------------------------- #


def test_reset_clears_everything(detector):
    t = warm(detector)
    detector.update(AD_AFTER_GAP, t)
    detector.reset()
    assert not detector.in_ad
    assert detector.baseline.count == 0
    decision = detector.update(AD_AFTER_GAP, t + WINDOW)
    assert decision.event is Event.NO_CHANGE  # baseline must be relearned


def test_reject_drops_ad_state_but_keeps_the_baseline(detector):
    t = warm(detector)
    detector.update(AD_AFTER_GAP, t)
    assert detector.in_ad
    detector.reject()
    assert not detector.in_ad
    assert detector.baseline.ready(5)
    # The next genuine seam can fire immediately — no baseline relearning.
    detector.update(CONTENT, t + WINDOW)
    decision = detector.update(AD_AFTER_GAP, t + 2 * WINDOW)
    assert decision.event is Event.AD_STARTED


def test_decision_metrics_expose_the_tuning_numbers(detector):
    t = warm(detector)
    decision = detector.update(AD_AFTER_GAP, t)
    assert decision.metrics["loudness_delta_db"] == pytest.approx(5.0)
    assert decision.metrics["crest_delta_db"] == pytest.approx(5.0)
    assert decision.metrics["baseline_rms_dbfs"] == pytest.approx(-20.0)
