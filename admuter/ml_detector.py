"""Phase 2: a per-window ad/content vote from a trained model.

This is deliberately *not* a Detector. It answers one question — "does this
window look like ad audio?" — and holds no state at all. Cue arming, hysteresis,
``min_ad_seconds`` and ``max_ad_seconds`` stay in the state machine, which is
where they can be reasoned about. A stateful model would duplicate that logic
somewhere it cannot be inspected, and the two copies would drift.

The vector is built here rather than by the caller because two of the four
features are differences against the live baseline, and their direction is a
silent failure: swap them and every prediction is wrong with no error anywhere.
The convention below matches ``HeuristicDetector._ad_profile`` exactly.

    loudness delta = window - baseline    (ads are LOUDER, so positive)
    crest delta    = baseline - window    (ads are FLATTER, so positive)

Both are "more positive means more ad-like", which is why they can share a sign
convention despite pointing opposite ways in raw terms.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Callable

from .detector import Baseline
from .features import Features

log = logging.getLogger(__name__)

# Bump when the saved dict's shape changes in a way older code cannot read.
SCHEMA_VERSION = 1


class MLDetectorError(RuntimeError):
    """Raised when a model file cannot be trusted to produce right answers."""


# The only columns this module knows how to compute at runtime. A model trained
# on anything else cannot be served here, and saying so at load time is the
# whole point: an unknown column silently filled with a zero, or a known one
# read out of order, produces confident nonsense.
FEATURE_BUILDERS: dict[str, Callable[[Features, Baseline], float]] = {
    "rms_dbfs": lambda f, b: f.rms_dbfs,
    "crest_db": lambda f, b: f.crest_db,
    # Sign conventions, matching _ad_profile(). Do not "tidy" these.
    "delta_rms_db": lambda f, b: f.rms_dbfs - float(b.rms_dbfs),
    "delta_crest_db": lambda f, b: float(b.crest_db) - f.crest_db,
}


def validate_columns(columns: object) -> list[str]:
    """Check a saved column list against what this module can actually build.

    Order is part of the contract, not an implementation detail: the scaler and
    the coefficients are positional. A model saved with the same four names in
    a different order will still load, still predict, and be wrong every time —
    so the list is preserved verbatim and used verbatim, never sorted or
    reconstructed from ``FEATURE_BUILDERS``.
    """
    if not isinstance(columns, (list, tuple)) or not columns:
        raise MLDetectorError(
            f"model has no usable feature column list (got {columns!r})"
        )
    if not all(isinstance(c, str) for c in columns):
        raise MLDetectorError(f"feature columns must all be strings: {columns!r}")
    unknown = [c for c in columns if c not in FEATURE_BUILDERS]
    if unknown:
        raise MLDetectorError(
            f"model needs feature(s) this build cannot compute: {unknown}. "
            f"Known: {sorted(FEATURE_BUILDERS)}"
        )
    duplicates = [c for c in set(columns) if list(columns).count(c) > 1]
    if duplicates:
        raise MLDetectorError(f"feature columns repeat: {sorted(duplicates)}")
    return list(columns)


def build_row(
    columns: list[str], features: Features, baseline: Baseline
) -> list[float]:
    """One feature vector, in the saved column order. Order is the contract."""
    return [FEATURE_BUILDERS[name](features, baseline) for name in columns]


def load_voter(detection) -> "MLVoter | None":
    """Build the voter a DetectionConfig asks for, or None. Never raises.

    A model that will not load must not take the service down: the heuristic
    alone is a working detector, and a stopped admuter cannot mute anything.
    Every failure here is logged at error level and swallowed -- including
    ImportError, since scikit-learn may simply not be installed on the box.

    The one thing it will not do is quietly disagree with the config. If the
    config says ml_vote_enabled and the model is unusable, that is said loudly,
    because the operator asked for an ensemble and is getting a heuristic.
    """
    if not getattr(detection, "ml_model_path", ""):
        return None
    try:
        voter = MLVoter.load(
            detection.ml_model_path,
            threshold=detection.ml_threshold,
            baseline_min_windows=detection.baseline_min_windows,
        )
    except Exception as exc:  # noqa: BLE001 - nothing here may reach the caller
        level = log.error if detection.ml_vote_enabled else log.warning
        level(
            "ML voter unavailable (%s); continuing with the heuristic alone%s",
            exc,
            " — detection.ml_vote_enabled is set but will have no effect"
            if detection.ml_vote_enabled else "",
        )
        return None
    log.info(
        "ML voter loaded from %s (threshold %.2f, %s)",
        detection.ml_model_path,
        detection.ml_threshold,
        "voting" if detection.ml_vote_enabled else "SHADOW MODE — logged, not counted",
    )
    return voter


class MLVoter:
    """A loaded model plus the glue that keeps its inputs honest.

    Not a Detector: no ``update``/``reset``/``reject``, no memory between
    windows. Feed it a window and the live baseline, get a vote back.
    """

    def __init__(
        self,
        pipeline: object,
        columns: list[str],
        threshold: float,
        baseline_min_windows: int,
        metadata: dict | None = None,
    ) -> None:
        self.pipeline = pipeline
        self.columns = validate_columns(columns)
        self.threshold = float(threshold)
        self.baseline_min_windows = int(baseline_min_windows)
        self.metadata = metadata or {}

    # ------------------------------------------------------------------ #
    # Loading
    # ------------------------------------------------------------------ #

    @classmethod
    def load(
        cls, path: str | Path, threshold: float, baseline_min_windows: int
    ) -> "MLVoter":
        """Read a ``train_model.py`` bundle, refusing anything suspect.

        Every check here is about a failure that would otherwise be silent.
        A model that loads and predicts confidently from the wrong columns is
        far worse than one that refuses to load.
        """
        import joblib  # local: the live service should not pay for this import
                       # unless a model is actually configured.

        from .config import resolve_path

        path = resolve_path(path)
        if not path.exists():
            raise MLDetectorError(f"no model at {path}")
        try:
            bundle = joblib.load(path)
        except Exception as exc:  # noqa: BLE001 - joblib raises many things
            raise MLDetectorError(f"could not load {path}: {exc}") from exc

        if not isinstance(bundle, dict):
            raise MLDetectorError(
                f"{path} is not a model bundle (got {type(bundle).__name__})"
            )
        version = bundle.get("schema_version")
        if version != SCHEMA_VERSION:
            raise MLDetectorError(
                f"{path} has schema_version {version!r}, this build reads "
                f"{SCHEMA_VERSION}"
            )
        pipeline = bundle.get("pipeline")
        if pipeline is None or not hasattr(pipeline, "predict_proba"):
            raise MLDetectorError(f"{path} has no pipeline with predict_proba")

        trained_with = bundle.get("sklearn_version")
        try:
            import sklearn

            if trained_with and trained_with != sklearn.__version__:
                # Not fatal — pickles usually survive a patch bump — but a
                # silent behaviour change is exactly what we are guarding
                # against elsewhere, so it gets said out loud.
                log.warning(
                    "model %s was trained with scikit-learn %s, running %s",
                    path.name, trained_with, sklearn.__version__,
                )
        except ImportError:  # pragma: no cover - sklearn is a hard dep of joblib use
            pass

        return cls(
            pipeline=pipeline,
            columns=bundle.get("feature_columns"),
            threshold=threshold,
            baseline_min_windows=baseline_min_windows,
            metadata={
                k: v for k, v in bundle.items() if k not in ("pipeline",)
            },
        )

    # ------------------------------------------------------------------ #
    # Voting
    # ------------------------------------------------------------------ #

    def says_ad(self, features: Features, baseline: Baseline) -> tuple[bool, float]:
        """(above_threshold, probability) for this one window.

        Returns ``(False, 0.0)`` whenever the answer would be a guess: silence,
        or a baseline too young for its deltas to mean anything. Half the
        feature set is measured against that baseline, so before it settles the
        model is being fed two columns of noise. Abstaining costs a missed ad;
        guessing costs muted dialogue.
        """
        if features.is_silence:
            return False, 0.0
        if not baseline.ready(self.baseline_min_windows):
            return False, 0.0
        if baseline.rms_dbfs is None or baseline.crest_db is None:
            return False, 0.0

        row = build_row(self.columns, features, baseline)
        probability = float(self.pipeline.predict_proba([row])[0][1])
        return probability >= self.threshold, probability
