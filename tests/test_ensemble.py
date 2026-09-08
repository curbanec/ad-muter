"""The two voters against one state machine.

These pin the three claims that make the ensemble safe to ship:

* shadow mode is inert — the model's opinion is recorded and cannot move a
  decision, so a new model can ride along on a live stream before anyone
  trusts it;
* entering an ad needs both voters, so the model can veto a mute the heuristic
  would have made on its own;
* staying in one needs either, so the model can hold a mute the heuristic was
  about to drop.

The asymmetry is the whole design: two votes to start muting, one to keep
muting. It mirrors the loudness hysteresis it sits next to.
"""

from __future__ import annotations

import dataclasses

import pytest

from admuter.config import DetectionConfig
from admuter.detector import Event, HeuristicDetector

from test_detector import AD, AD_AFTER_GAP, CONTENT, WINDOW, feed, warm


def events(detector, window, count, start):
    """Every event over `count` windows. feed() only returns the last one, and
    AD_ENDED fires on the window that completes the streak, not after it."""
    out = []
    for i in range(count):
        out.append(detector.update(window, start + i * WINDOW).event)
    return out


class FakeVoter:
    """Says whatever it is told to, and counts how often it was consulted."""

    def __init__(self, verdict: bool = True, probability: float | None = None) -> None:
        self.verdict = verdict
        self.probability = probability if probability is not None else (
            0.99 if verdict else 0.01
        )
        self.calls = 0

    def says_ad(self, features, baseline):
        self.calls += 1
        return self.verdict, self.probability


def build(verdict: bool, *, enabled: bool, **overrides) -> tuple:
    config = DetectionConfig(
        baseline_min_windows=5,
        ml_model_path="fake.joblib" if enabled else "",
        ml_vote_enabled=enabled,
        **overrides,
    )
    voter = FakeVoter(verdict)
    return HeuristicDetector(config, WINDOW, ml_voter=voter), voter


# --------------------------------------------------------------------------- #
# No voter at all
# --------------------------------------------------------------------------- #


def test_without_a_voter_no_ml_keys_are_logged():
    """An install that never heard of Phase 2 logs exactly what it logged before."""
    detector = HeuristicDetector(DetectionConfig(baseline_min_windows=5), WINDOW)
    t = warm(detector)
    decision = detector.update(AD_AFTER_GAP, t)
    for key in ("ml_probability", "ml_says_ad", "heuristic_says_ad", "voters_disagree"):
        assert key not in decision.metrics


# --------------------------------------------------------------------------- #
# Shadow mode
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("verdict", [True, False])
def test_shadow_mode_never_changes_the_decision(verdict):
    """Whatever the model says, the events must match the heuristic alone."""
    plain = HeuristicDetector(DetectionConfig(baseline_min_windows=5), WINDOW)
    shadowed, voter = build(verdict, enabled=False)

    sequence = [CONTENT] * 8 + [AD_AFTER_GAP] + [AD] * 6 + [CONTENT] * 10
    plain_events, shadow_events = [], []
    for i, window in enumerate(sequence):
        t = i * WINDOW
        plain_events.append(plain.update(window, t).event)
        shadow_events.append(shadowed.update(window, t).event)

    assert shadow_events == plain_events
    assert shadowed.in_ad == plain.in_ad
    assert voter.calls > 0, "shadow mode must still consult the model"


def test_shadow_mode_records_the_opinion_it_ignores():
    detector, _ = build(True, enabled=False)
    t = warm(detector)
    decision = detector.update(CONTENT, t)  # heuristic says content, model says ad
    assert decision.metrics["ml_says_ad"] == 1.0
    assert decision.metrics["heuristic_says_ad"] == 0.0
    assert decision.metrics["voters_disagree"] == 1.0
    assert decision.metrics["ml_probability"] == pytest.approx(0.99)
    assert decision.event is Event.NO_CHANGE
    assert not detector.in_ad


# --------------------------------------------------------------------------- #
# AND to enter
# --------------------------------------------------------------------------- #


def test_a_disagreeing_model_blocks_an_entry_the_heuristic_would_have_made():
    baseline_detector = HeuristicDetector(DetectionConfig(baseline_min_windows=5), WINDOW)
    t = warm(baseline_detector)
    assert baseline_detector.update(AD_AFTER_GAP, t).event is Event.AD_STARTED

    vetoed, voter = build(False, enabled=True)
    t = warm(vetoed)
    decision = vetoed.update(AD_AFTER_GAP, t)
    assert decision.event is Event.NO_CHANGE
    assert not vetoed.in_ad
    assert decision.metrics["heuristic_says_ad"] == 1.0
    assert decision.metrics["ml_says_ad"] == 0.0
    assert voter.calls > 0


def test_an_agreeing_model_leaves_the_entry_alone():
    detector, _ = build(True, enabled=True)
    t = warm(detector)
    assert detector.update(AD_AFTER_GAP, t).event is Event.AD_STARTED
    assert detector.in_ad


def test_the_model_alone_cannot_start_an_ad():
    """AND means the model is a veto, never a trigger."""
    detector, _ = build(True, enabled=True)
    decision, _ = feed(detector, CONTENT, 30)
    assert decision.event is Event.NO_CHANGE
    assert not detector.in_ad


# --------------------------------------------------------------------------- #
# OR to stay
# --------------------------------------------------------------------------- #


def test_the_model_extends_a_mute_the_heuristic_would_have_dropped():
    """Once in an ad, either voter can hold it open."""
    config = dict(baseline_min_windows=5, ad_end_windows=2, min_ad_seconds=0.0)

    plain = HeuristicDetector(DetectionConfig(**config), WINDOW)
    t = warm(plain)
    assert plain.update(AD_AFTER_GAP, t).event is Event.AD_STARTED
    # Content-shaped windows: the heuristic's profile goes false and the ad ends.
    assert Event.AD_ENDED in events(plain, CONTENT, 4, t + WINDOW)
    assert not plain.in_ad

    held, _ = build(True, enabled=True, ad_end_windows=2, min_ad_seconds=0.0)
    t = warm(held)
    assert held.update(AD_AFTER_GAP, t).event is Event.AD_STARTED
    assert Event.AD_ENDED not in events(held, CONTENT, 4, t + WINDOW)
    assert held.in_ad, "the model's vote should have kept the ad alive"

    last = held.update(CONTENT, t + 5 * WINDOW)
    assert last.metrics["heuristic_says_ad"] == 0.0
    assert last.metrics["ml_says_ad"] == 1.0


def test_a_silent_model_does_not_prolong_an_ad():
    """OR-to-stay must not become 'never end'."""
    detector, _ = build(False, enabled=True, ad_end_windows=2, min_ad_seconds=0.0)
    t = warm(detector)
    # The model vetoes entry, so drive the state machine in without it.
    detector._ml_voter = None
    assert detector.update(AD_AFTER_GAP, t).event is Event.AD_STARTED
    detector._ml_voter = FakeVoter(False)
    assert Event.AD_ENDED in events(detector, CONTENT, 4, t + WINDOW)
    assert not detector.in_ad


# --------------------------------------------------------------------------- #
# Failure handling
# --------------------------------------------------------------------------- #


def test_a_missing_model_leaves_the_detector_working(caplog):
    from admuter.ml_detector import load_voter

    config = DetectionConfig(
        baseline_min_windows=5, ml_model_path="/nonexistent/model.joblib"
    )
    with caplog.at_level("WARNING"):
        assert load_voter(config) is None
    assert "heuristic alone" in caplog.text

    detector = HeuristicDetector(config, WINDOW, ml_voter=load_voter(config))
    t = warm(detector)
    assert detector.update(AD_AFTER_GAP, t).event is Event.AD_STARTED


def test_an_enabled_but_unloadable_model_is_an_error_not_a_crash(caplog):
    from admuter.ml_detector import load_voter

    config = DetectionConfig(
        ml_model_path="/nonexistent/model.joblib", ml_vote_enabled=True
    )
    with caplog.at_level("ERROR"):
        assert load_voter(config) is None
    assert "will have no effect" in caplog.text


def test_no_path_means_no_voter_and_no_import():
    from admuter.ml_detector import load_voter

    assert load_voter(DetectionConfig()) is None


def test_enabling_the_vote_without_a_model_is_a_config_error():
    from admuter.config import ConfigError

    with pytest.raises(ConfigError, match="no model to vote with"):
        DetectionConfig(ml_vote_enabled=True).validate()
