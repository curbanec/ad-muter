"""Guards for the two ways the ML voter fails silently.

Both failures here produce confident, wrong predictions with no exception and
no log line: a feature vector assembled in the wrong order, and a delta whose
sign is inverted. Neither is visible in a score — the model just gets worse —
so they are pinned by tests rather than left to review.
"""

from __future__ import annotations

import pytest

from admuter.detector import Baseline
from admuter.features import Features
from admuter.ml_detector import (
    FEATURE_BUILDERS,
    SCHEMA_VERSION,
    MLDetectorError,
    MLVoter,
    build_row,
    validate_columns,
)


def features(rms_dbfs: float = -20.0, crest_db: float = 10.0, **kwargs) -> Features:
    defaults = dict(
        rms_dbfs=rms_dbfs,
        peak=0.5,
        peak_dbfs=-6.0,
        crest_factor=3.0,
        crest_db=crest_db,
        spectral_centroid_hz=2000.0,
        zero_crossing_rate=0.05,
        is_silence=False,
        silence_ratio=0.0,
        max_silence_run_seconds=0.0,
        leading_silence_seconds=0.0,
        trailing_silence_seconds=0.0,
        interior_silence_seconds=0.0,
        duration_seconds=1.0,
    )
    defaults.update(kwargs)
    return Features(**defaults)


def baseline(rms_dbfs: float = -28.0, crest_db: float = 14.0, count: int = 100):
    return Baseline(rms_dbfs=rms_dbfs, crest_db=crest_db, centroid_hz=1800.0, count=count)


class Always:
    """Stand-in pipeline: returns a fixed probability, records what it saw."""

    def __init__(self, probability: float = 0.9) -> None:
        self.probability = probability
        self.seen: list[list[float]] = []

    def predict_proba(self, rows):
        self.seen.extend([list(r) for r in rows])
        return [[1.0 - self.probability, self.probability] for _ in rows]


# --------------------------------------------------------------------------- #
# Column order and identity
# --------------------------------------------------------------------------- #


def test_unknown_column_is_rejected():
    with pytest.raises(MLDetectorError, match="cannot compute"):
        validate_columns(["rms_dbfs", "spectral_centroid_hz"])


def test_repeated_column_is_rejected():
    with pytest.raises(MLDetectorError, match="repeat"):
        validate_columns(["rms_dbfs", "rms_dbfs"])


@pytest.mark.parametrize("bad", [[], None, "rms_dbfs", [1, 2]])
def test_malformed_column_lists_are_rejected(bad):
    with pytest.raises(MLDetectorError):
        validate_columns(bad)


def test_build_row_follows_the_saved_order_not_a_canonical_one():
    """The scaler and coefficients are positional, so order is the contract."""
    f, b = features(rms_dbfs=-20.0, crest_db=10.0), baseline(-28.0, 14.0)
    forward = build_row(["rms_dbfs", "crest_db"], f, b)
    reversed_ = build_row(["crest_db", "rms_dbfs"], f, b)
    assert forward == [-20.0, 10.0]
    assert reversed_ == [10.0, -20.0]


def test_voter_preserves_column_order_into_the_pipeline():
    pipeline = Always()
    voter = MLVoter(
        pipeline=pipeline,
        columns=["delta_crest_db", "rms_dbfs"],  # deliberately not the usual order
        threshold=0.5,
        baseline_min_windows=30,
    )
    voter.says_ad(features(rms_dbfs=-20.0, crest_db=10.0), baseline(-28.0, 14.0))
    # delta_crest_db = 14 - 10 = 4.0, then rms_dbfs = -20.0, in that order.
    assert pipeline.seen == [[4.0, -20.0]]


def test_every_known_builder_is_exercised_by_the_saved_default():
    """The trainer's column list must stay inside what the voter can build."""
    from scripts.train_model import FEATURE_COLUMNS

    assert set(FEATURE_COLUMNS) <= set(FEATURE_BUILDERS)
    assert validate_columns(FEATURE_COLUMNS) == FEATURE_COLUMNS


# --------------------------------------------------------------------------- #
# Delta sign convention
# --------------------------------------------------------------------------- #


def test_loudness_delta_is_window_minus_baseline():
    """Ads are LOUDER than the baseline, so a louder window must read positive."""
    row = build_row(["delta_rms_db"], features(rms_dbfs=-20.0), baseline(rms_dbfs=-28.0))
    assert row == [8.0]


def test_crest_delta_is_baseline_minus_window():
    """Ads are FLATTER than the baseline, so a squashed window must read positive."""
    row = build_row(["delta_crest_db"], features(crest_db=10.0), baseline(crest_db=14.0))
    assert row == [4.0]


def test_the_two_deltas_do_not_share_a_direction():
    """Both are 'more positive is more ad-like' despite opposite raw senses.

    Swapping either subtraction still produces plausible-looking numbers, which
    is exactly why this is pinned: an ad-like window must score positive on
    both, and a content-like window negative on both.
    """
    ad_like = build_row(
        ["delta_rms_db", "delta_crest_db"],
        features(rms_dbfs=-18.0, crest_db=9.0),   # louder and flatter
        baseline(rms_dbfs=-28.0, crest_db=14.0),
    )
    content_like = build_row(
        ["delta_rms_db", "delta_crest_db"],
        features(rms_dbfs=-33.0, crest_db=17.0),  # quieter and more dynamic
        baseline(rms_dbfs=-28.0, crest_db=14.0),
    )
    assert all(v > 0 for v in ad_like), ad_like
    assert all(v < 0 for v in content_like), content_like


def test_detector_and_voter_agree_on_the_delta_convention():
    """Pin the voter to HeuristicDetector._ad_profile rather than to a comment."""
    from admuter.config import DetectionConfig
    from admuter.detector import HeuristicDetector

    f, b = features(rms_dbfs=-18.0, crest_db=9.0), baseline(rms_dbfs=-28.0, crest_db=14.0)
    detector = HeuristicDetector(DetectionConfig(), window_seconds=1.0)
    detector.baseline = b
    _, metrics = detector._ad_profile(f, loudness_threshold_db=0.0, smoothed_rms_dbfs=-18.0)

    row = build_row(["delta_rms_db", "delta_crest_db"], f, b)
    assert row == [metrics["loudness_delta_db"], metrics["crest_delta_db"]]


# --------------------------------------------------------------------------- #
# Abstaining
# --------------------------------------------------------------------------- #


def test_cold_baseline_abstains_rather_than_guessing():
    voter = MLVoter(Always(0.99), ["rms_dbfs"], threshold=0.5, baseline_min_windows=30)
    assert voter.says_ad(features(), baseline(count=29)) == (False, 0.0)
    assert voter.says_ad(features(), baseline(count=30))[0] is True


def test_silence_abstains():
    voter = MLVoter(Always(0.99), ["rms_dbfs"], threshold=0.5, baseline_min_windows=30)
    assert voter.says_ad(features(is_silence=True), baseline()) == (False, 0.0)


def test_unset_baseline_values_abstain_even_when_count_is_high():
    voter = MLVoter(Always(0.99), ["rms_dbfs"], threshold=0.5, baseline_min_windows=1)
    empty = Baseline(count=99)  # counted, but never given a value
    assert voter.says_ad(features(), empty) == (False, 0.0)


# --------------------------------------------------------------------------- #
# Load-time refusal
# --------------------------------------------------------------------------- #


def bundle(**overrides) -> dict:
    base = {
        "schema_version": SCHEMA_VERSION,
        "pipeline": Always(),
        "feature_columns": ["rms_dbfs", "crest_db"],
        "sklearn_version": "0.0.0-test",
    }
    base.update(overrides)
    return base


def dump(tmp_path, payload):
    import joblib

    path = tmp_path / "model.joblib"
    joblib.dump(payload, path)
    return path


def test_load_rejects_a_stale_schema(tmp_path):
    path = dump(tmp_path, bundle(schema_version=SCHEMA_VERSION + 1))
    with pytest.raises(MLDetectorError, match="schema_version"):
        MLVoter.load(path, threshold=0.5, baseline_min_windows=30)


def test_load_rejects_columns_this_build_cannot_compute(tmp_path):
    path = dump(tmp_path, bundle(feature_columns=["rms_dbfs", "seconds_since_gap"]))
    with pytest.raises(MLDetectorError, match="cannot compute"):
        MLVoter.load(path, threshold=0.5, baseline_min_windows=30)


def test_load_rejects_a_bundle_without_a_pipeline(tmp_path):
    path = dump(tmp_path, bundle(pipeline=None))
    with pytest.raises(MLDetectorError, match="predict_proba"):
        MLVoter.load(path, threshold=0.5, baseline_min_windows=30)


def test_load_rejects_a_missing_file(tmp_path):
    with pytest.raises(MLDetectorError, match="no model at"):
        MLVoter.load(tmp_path / "absent.joblib", threshold=0.5, baseline_min_windows=30)


def test_load_round_trips_the_real_column_order(tmp_path):
    from scripts.train_model import FEATURE_COLUMNS

    path = dump(tmp_path, bundle(feature_columns=list(FEATURE_COLUMNS)))
    voter = MLVoter.load(path, threshold=0.5, baseline_min_windows=30)
    assert voter.columns == FEATURE_COLUMNS
