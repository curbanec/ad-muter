"""SeamDoorDetector: judge at the joins, say nothing in between.

Each test drives one behaviour with synthetic Features. No audio, no clock —
timestamps are arguments, so these run identically to the live service.
"""

from __future__ import annotations

import dataclasses

import pytest

from admuter.config import ConfigError, DetectionConfig
from admuter.detector import Event
from admuter.seam_door import SeamDoorDetector, build_detector

from test_detector import WINDOW, make_features

SEAM = dict(interior=0.4, silence_ratio=0.4)   # a qualifying gap inside a window


def config(**overrides) -> DetectionConfig:
    base = dict(
        mode="seam_door",
        door_before_seconds=10.0,
        door_after_seconds=3.0,
        door_enter_step_db=3.0,
        door_exit_margin_db=2.0,
        door_fallback_seconds=15.0,
        min_ad_seconds=5.0,
        max_ad_seconds=200.0,
    )
    base.update(overrides)
    return DetectionConfig(**base)


def run(detector, windows, start=0.0):
    """Feed (features, count) pairs; return every decision."""
    out, t = [], start
    for features, count in windows:
        for _ in range(count):
            out.append((t, detector.update(features, t)))
            t += WINDOW
    return out


def show(level=-30.0):
    return make_features(rms_dbfs=level)


def seam_then(level, count=4):
    """A seam window at `level`, then more of the same."""
    return [(make_features(rms_dbfs=level, **SEAM), 1),
            (make_features(rms_dbfs=level), count)]


def settle(detector, level=-30.0, seconds=12):
    """Fill the before-buffer with show audio."""
    run(detector, [(show(level), seconds)])
    return seconds * WINDOW


# --------------------------------------------------------------------------- #
# Entering
# --------------------------------------------------------------------------- #


def test_enters_on_a_loud_step_at_a_seam():
    detector = SeamDoorDetector(config(), WINDOW)
    t = settle(detector)
    events = [d.event for _, d in run(detector, seam_then(-20.0), start=t)]
    assert Event.AD_STARTED in events
    assert detector.in_ad


def test_does_not_enter_on_a_small_step():
    """+2 dB against a 3 dB bar is a door onto the same room."""
    detector = SeamDoorDetector(config(), WINDOW)
    t = settle(detector)
    events = [d.event for _, d in run(detector, seam_then(-28.0), start=t)]
    assert Event.AD_STARTED not in events
    assert not detector.in_ad


def test_does_not_enter_when_the_audio_gets_quieter():
    detector = SeamDoorDetector(config(), WINDOW)
    t = settle(detector)
    events = [d.event for _, d in run(detector, seam_then(-40.0), start=t)]
    assert Event.AD_STARTED not in events


def test_never_enters_without_a_seam():
    """Loud audio alone is not a door; the join is what makes it one."""
    detector = SeamDoorDetector(config(), WINDOW)
    t = settle(detector)
    events = [d.event for _, d in run(detector, [(show(-15.0), 30)], start=t)]
    assert Event.AD_STARTED not in events


def test_the_step_is_measured_against_show_level_not_the_seam_window():
    detector = SeamDoorDetector(config(), WINDOW)
    t = settle(detector, level=-30.0)
    decisions = run(detector, seam_then(-20.0), start=t)
    started = next(d for _, d in decisions if d.event is Event.AD_STARTED)
    assert started.metrics["before_level"] == pytest.approx(-30.0, abs=0.1)
    assert started.metrics["after_level"] == pytest.approx(-20.0, abs=0.1)
    assert started.metrics["step"] == pytest.approx(10.0, abs=0.1)


# --------------------------------------------------------------------------- #
# Staying
# --------------------------------------------------------------------------- #


def test_a_loud_to_loud_seam_inside_an_ad_does_not_exit():
    """Spot-to-spot joins are the common case and must change nothing."""
    detector = SeamDoorDetector(config(), WINDOW)
    t = settle(detector)
    run(detector, seam_then(-20.0), start=t)
    assert detector.in_ad

    later = run(detector, seam_then(-19.0), start=t + 10 * WINDOW)
    assert Event.AD_ENDED not in [d.event for _, d in later]
    assert detector.in_ad
    assert any("still loud" in d.reason for _, d in later)


def test_quiet_windows_inside_an_ad_do_not_exit_without_a_seam():
    """No re-judging between seams: that is what stops the flicker."""
    detector = SeamDoorDetector(config(), WINDOW)
    t = settle(detector)
    run(detector, seam_then(-20.0), start=t)
    # A few quiet windows, fewer than door_fallback_seconds.
    later = run(detector, [(show(-45.0), 5)], start=t + 10 * WINDOW)
    assert Event.AD_ENDED not in [d.event for _, d in later]
    assert detector.in_ad


# --------------------------------------------------------------------------- #
# Leaving
# --------------------------------------------------------------------------- #


def test_exits_at_a_seam_back_to_show_level():
    detector = SeamDoorDetector(config(min_ad_seconds=0.0), WINDOW)
    t = settle(detector, level=-30.0)
    run(detector, seam_then(-20.0), start=t)
    assert detector.in_ad

    out = run(detector, seam_then(-30.0), start=t + 20 * WINDOW)
    assert Event.AD_ENDED in [d.event for _, d in out]
    assert not detector.in_ad


def test_min_ad_seconds_blocks_an_early_exit():
    detector = SeamDoorDetector(config(min_ad_seconds=60.0), WINDOW)
    t = settle(detector, level=-30.0)
    run(detector, seam_then(-20.0), start=t)
    out = run(detector, seam_then(-30.0), start=t + 10 * WINDOW)
    assert Event.AD_ENDED not in [d.event for _, d in out]
    assert detector.in_ad


def test_the_fallback_exit_fires_without_a_seam():
    """Covers a missed exit seam, which coverage says happens 4 times in 25."""
    detector = SeamDoorDetector(
        config(min_ad_seconds=0.0, door_fallback_seconds=5.0), WINDOW
    )
    t = settle(detector, level=-30.0)
    run(detector, seam_then(-20.0), start=t)
    assert detector.in_ad

    out = run(detector, [(show(-31.0), 12)], start=t + 20 * WINDOW)
    ended = [d for _, d in out if d.event is Event.AD_ENDED]
    assert ended, "a sustained return to show level should end the ad"
    assert "fallback" in ended[0].reason


def test_the_failsafe_fires():
    detector = SeamDoorDetector(config(max_ad_seconds=30.0), WINDOW)
    t = settle(detector, level=-30.0)
    run(detector, seam_then(-20.0), start=t)
    out = run(detector, [(show(-20.0), 3)], start=t + 100 * WINDOW)
    ended = [d for _, d in out if d.event is Event.AD_ENDED]
    assert ended and "failsafe" in ended[0].reason
    assert not detector.in_ad


# --------------------------------------------------------------------------- #
# Collection bookkeeping
# --------------------------------------------------------------------------- #


def test_a_second_seam_during_collection_keeps_the_earlier_before_level():
    """The sliver between two close seams is not representative show audio."""
    detector = SeamDoorDetector(config(), WINDOW)
    t = settle(detector, level=-30.0)

    # Seam, one loud window, then a second seam before collection finishes.
    decisions = run(detector, [
        (make_features(rms_dbfs=-20.0, **SEAM), 1),
        (make_features(rms_dbfs=-20.0), 1),
        (make_features(rms_dbfs=-20.0, **SEAM), 1),
        (make_features(rms_dbfs=-20.0), 3),
    ], start=t)
    started = [d for _, d in decisions if d.event is Event.AD_STARTED]
    assert started, "the restarted collection should still complete"
    # -30 is show level; had the restart taken its before from the loud sliver
    # the step would have been ~0 and nothing would have fired.
    assert started[0].metrics["before_level"] == pytest.approx(-30.0, abs=0.1)


def test_reject_keeps_the_before_buffer_and_reset_clears_it():
    detector = SeamDoorDetector(config(), WINDOW)
    t = settle(detector, level=-30.0)
    run(detector, seam_then(-20.0), start=t)
    assert detector.in_ad

    detector.reject()
    assert not detector.in_ad
    assert detector._before, "the last ten seconds of television are still true"

    detector.reset()
    assert not detector._before


def test_reject_drops_a_pending_collection():
    detector = SeamDoorDetector(config(), WINDOW)
    t = settle(detector)
    run(detector, [(make_features(rms_dbfs=-20.0, **SEAM), 1)], start=t)
    assert detector._pending_before is not None
    detector.reject()
    assert detector._pending_before is None


def test_ad_profile_is_true_exactly_while_in_an_ad():
    """The controller counts this to confirm, so it must track the state.

    Checked window by window rather than after the fact: comparing a whole
    run of decisions against the detector's final state proves nothing.
    """
    detector = SeamDoorDetector(config(min_ad_seconds=0.0), WINDOW)
    t = settle(detector, level=-30.0)
    sequence = [make_features(rms_dbfs=-20.0, **SEAM)] + [
        make_features(rms_dbfs=-20.0) for _ in range(4)
    ]
    saw_ad = False
    for i, features in enumerate(sequence):
        decision = detector.update(features, t + i * WINDOW)
        assert decision.ad_profile == detector.in_ad
        saw_ad = saw_ad or detector.in_ad
    assert saw_ad, "the sequence never entered an ad, so nothing was tested"


# --------------------------------------------------------------------------- #
# Mode switching
# --------------------------------------------------------------------------- #


def test_legacy_is_the_default_and_builds_the_heuristic():
    from admuter.detector import HeuristicDetector

    assert isinstance(build_detector(DetectionConfig(), WINDOW), HeuristicDetector)


def test_seam_door_mode_builds_the_seam_door_detector():
    assert isinstance(
        build_detector(DetectionConfig(mode="seam_door"), WINDOW), SeamDoorDetector
    )


def test_seam_door_ignores_the_voters_it_is_handed():
    class Loud:
        def says_ad(self, *a, **k):
            return True, 1.0

    detector = build_detector(
        DetectionConfig(mode="seam_door"), WINDOW, ml_voter=Loud()
    )
    assert not hasattr(detector, "_ml_voter")


def test_an_unknown_mode_is_a_config_error():
    with pytest.raises(ConfigError, match="detection.mode"):
        DetectionConfig(mode="magic").validate()
