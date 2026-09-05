"""Feature history -> transition events.

Phase 1 is deliberately rule-based. The interface (`Detector`) is the seam a
Phase 2 ML classifier drops into: as long as something accepts a `Features` plus
a timestamp and returns a `Decision`, the controller does not care how the call
is implemented.

Bias throughout: when the evidence is thin, emit NO_CHANGE. A missed ad is
annoying; muting real dialogue is worse.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Protocol, runtime_checkable

from .config import DetectionConfig
from .features import Features


class Event(str, Enum):
    AD_STARTED = "AD_STARTED"
    AD_ENDED = "AD_ENDED"
    NO_CHANGE = "NO_CHANGE"


@dataclass(frozen=True)
class Decision:
    """What the detector concluded about one window.

    ``event`` is the edge (fires once); ``ad_profile`` is the level (true for
    every window that currently looks ad-like). The controller counts the latter
    to build confirmation, so a Phase 2 classifier must populate both.
    """

    event: Event
    timestamp: float
    ad_profile: bool
    confidence: float
    reason: str
    metrics: dict[str, float] = field(default_factory=dict)

    @property
    def is_change(self) -> bool:
        return self.event is not Event.NO_CHANGE


@runtime_checkable
class Detector(Protocol):
    """Swap-in point for Phase 2."""

    def update(self, features: Features, timestamp: float) -> Decision:
        """Consume one window and return the resulting decision."""

    def reset(self) -> None:
        """Forget all history (capture restart, app change, service resume)."""

    def reject(self) -> None:
        """Controller declined to act on the last AD_STARTED; drop ad state."""


@dataclass
class Baseline:
    """Slow-moving picture of what 'content' sounds like on this stream."""

    rms_dbfs: float | None = None
    crest_db: float | None = None
    centroid_hz: float | None = None
    count: int = 0

    def update(self, features: Features, alpha: float) -> None:
        if self.count == 0:
            self.rms_dbfs = features.rms_dbfs
            self.crest_db = features.crest_db
            self.centroid_hz = features.spectral_centroid_hz
        else:
            self.rms_dbfs = _ema(self.rms_dbfs, features.rms_dbfs, alpha)
            self.crest_db = _ema(self.crest_db, features.crest_db, alpha)
            self.centroid_hz = _ema(
                self.centroid_hz, features.spectral_centroid_hz, alpha
            )
        self.count += 1

    def ready(self, min_windows: int) -> bool:
        return self.count >= min_windows

    def as_dict(self) -> dict[str, float]:
        return {
            "baseline_rms_dbfs": _or_nan(self.rms_dbfs),
            "baseline_crest_db": _or_nan(self.crest_db),
            "baseline_centroid_hz": _or_nan(self.centroid_hz),
            "baseline_count": float(self.count),
        }


def _ema(previous: float | None, value: float, alpha: float) -> float:
    if previous is None:
        return value
    return (1.0 - alpha) * previous + alpha * value


def _or_nan(value: float | None) -> float:
    return float("nan") if value is None else float(value)


def _clamp01(value: float) -> float:
    return max(0.0, min(1.0, value))


class HeuristicDetector:
    """Rule-based Phase 1 detector.

    Two pieces of evidence must line up before an ad is declared:

    1. **Transition cue** — a short near-silent gap (0.2–1.5 s by default)
       followed by an abrupt loudness or spectral shift. Netflix's server-side
       stitching leaves this seam even though the Roku reports nothing.
    2. **Ad profile** — sustained loudness above the content baseline *and* a
       crest factor below it (ads are compressed harder than show audio).

    The cue stays armed for a short grace period, so the profile does not have
    to be conclusive in the very first window after the seam.

    The loudness half of the profile has hysteresis: entering an ad needs
    ``ad_loudness_delta_db`` over the baseline, but staying in one only needs
    the lower ``ad_stay_loudness_delta_db``, so a quieter spot mid-break does
    not end the ad early. The crest test is the same in both states.
    """

    def __init__(self, config: DetectionConfig, window_seconds: float = 1.0) -> None:
        self.config = config
        self.window_seconds = window_seconds
        self.baseline = Baseline()
        self._in_ad = False
        self._ad_started_at: float | None = None
        self._carry_silence = 0.0
        self._cue_at: float | None = None
        self._cue_gap = 0.0
        self._last_voiced: Features | None = None
        self._non_ad_streak = 0

    # ------------------------------------------------------------------ #
    # Detector protocol
    # ------------------------------------------------------------------ #

    def reset(self) -> None:
        """Full reset, including the learned baseline."""
        self.baseline = Baseline()
        self.reject()
        self._carry_silence = 0.0
        self._last_voiced = None

    def reject(self) -> None:
        """Clear ad state but keep the baseline we worked to learn."""
        self._in_ad = False
        self._ad_started_at = None
        self._cue_at = None
        self._cue_gap = 0.0
        self._non_ad_streak = 0

    @property
    def in_ad(self) -> bool:
        return self._in_ad

    def update(self, features: Features, timestamp: float) -> Decision:
        cfg = self.config
        gap = self._track_silence(features)
        cue_gap = self._maybe_arm_cue(features, timestamp, gap)

        # Hysteresis: the bar the loudness delta must clear depends on state.
        loudness_threshold = (
            cfg.ad_stay_loudness_delta_db if self._in_ad else cfg.ad_loudness_delta_db
        )
        profile, profile_metrics = self._ad_profile(features, loudness_threshold)
        cue_active = self._cue_active(timestamp)

        metrics: dict[str, float] = {
            "rms_dbfs": features.rms_dbfs,
            "crest_db": features.crest_db,
            "centroid_hz": features.spectral_centroid_hz,
            "gap_seconds": gap,
            "cue_gap_seconds": cue_gap,
            "cue_active": float(cue_active),
            "ad_loudness_delta_db": cfg.ad_loudness_delta_db,
            "ad_stay_loudness_delta_db": cfg.ad_stay_loudness_delta_db,
            **profile_metrics,
            **self.baseline.as_dict(),
        }

        if not features.is_silence:
            self._last_voiced = features

        if self._in_ad:
            return self._update_in_ad(features, timestamp, profile, metrics)
        return self._update_in_content(features, timestamp, profile, cue_active, metrics)

    # ------------------------------------------------------------------ #
    # Internals
    # ------------------------------------------------------------------ #

    def _track_silence(self, features: Features) -> float:
        """Stitch silent runs across window boundaries; return a completed gap.

        Returns the length in seconds of a silent stretch that ended inside this
        window (0.0 if none ended here). A run still in progress at the window
        edge is carried forward instead of being reported early.
        """
        fully_silent = features.silence_ratio >= 0.999
        if fully_silent:
            self._carry_silence += features.duration_seconds
            return 0.0

        completed = max(
            self._carry_silence + features.leading_silence_seconds,
            features.interior_silence_seconds,
        )
        self._carry_silence = features.trailing_silence_seconds
        return completed

    def _maybe_arm_cue(
        self, features: Features, timestamp: float, gap: float
    ) -> float:
        """Arm the transition cue if a qualifying gap + abrupt shift occurred."""
        cfg = self.config
        if not cfg.require_transition_cue:
            self._cue_at = timestamp
            self._cue_gap = gap
            return gap
        if not (cfg.min_gap_seconds <= gap <= cfg.max_gap_seconds):
            return 0.0

        before = self._last_voiced
        loudness_jump = (
            abs(features.rms_dbfs - before.rms_dbfs) if before is not None else 0.0
        )
        centroid_shift = 0.0
        if before is not None and before.spectral_centroid_hz > 1.0:
            centroid_shift = abs(
                features.spectral_centroid_hz - before.spectral_centroid_hz
            ) / before.spectral_centroid_hz

        abrupt = (
            before is None
            or loudness_jump >= cfg.loudness_jump_db
            or centroid_shift >= cfg.centroid_shift_ratio
        )
        if not abrupt:
            return 0.0

        self._cue_at = timestamp
        self._cue_gap = gap
        return gap

    def _cue_active(self, timestamp: float) -> bool:
        if self._cue_at is None:
            return False
        if timestamp - self._cue_at > self.config.cue_grace_seconds:
            self._cue_at = None
            return False
        return True

    def _ad_profile(
        self, features: Features, loudness_threshold_db: float
    ) -> tuple[bool, dict[str, float]]:
        """Does this window look like ad audio next to the content baseline?

        ``loudness_threshold_db`` is the bar the loudness delta must clear:
        ``ad_loudness_delta_db`` to enter an ad, ``ad_stay_loudness_delta_db``
        to remain in one. The crest test does not change between the two.
        """
        cfg = self.config
        if features.is_silence or not self.baseline.ready(cfg.baseline_min_windows):
            return False, {"loudness_delta_db": 0.0, "crest_delta_db": 0.0}

        loudness_delta = features.rms_dbfs - float(self.baseline.rms_dbfs)
        crest_delta = float(self.baseline.crest_db) - features.crest_db
        louder = loudness_delta >= loudness_threshold_db
        squashed = crest_delta >= cfg.ad_crest_delta_db
        return louder and squashed, {
            "loudness_delta_db": loudness_delta,
            "crest_delta_db": crest_delta,
        }

    def _confidence(self, metrics: dict[str, float]) -> float:
        cfg = self.config
        loud = _clamp01(
            metrics.get("loudness_delta_db", 0.0) / max(cfg.ad_loudness_delta_db, 1e-6)
        )
        crest = _clamp01(
            metrics.get("crest_delta_db", 0.0) / max(cfg.ad_crest_delta_db, 1e-6)
        )
        return round(_clamp01(0.5 * loud + 0.5 * crest), 3)

    def _update_in_content(
        self,
        features: Features,
        timestamp: float,
        profile: bool,
        cue_active: bool,
        metrics: dict[str, float],
    ) -> Decision:
        cfg = self.config
        if profile and cue_active:
            self._in_ad = True
            self._ad_started_at = timestamp
            self._cue_at = None
            self._non_ad_streak = 0
            return Decision(
                event=Event.AD_STARTED,
                timestamp=timestamp,
                ad_profile=True,
                confidence=self._confidence(metrics),
                reason=(
                    f"gap={metrics['cue_gap_seconds']:.2f}s then "
                    f"loudness +{metrics['loudness_delta_db']:.1f}dB / "
                    f"crest -{metrics['crest_delta_db']:.1f}dB vs baseline"
                ),
                metrics=metrics,
            )

        if not features.is_silence:
            self.baseline.update(features, cfg.baseline_alpha)

        if not self.baseline.ready(cfg.baseline_min_windows):
            reason = (
                f"baseline warming up ({self.baseline.count}/"
                f"{cfg.baseline_min_windows} windows)"
            )
        elif profile:
            reason = "ad profile without a transition cue — holding"
        elif cue_active:
            reason = "transition cue armed, waiting for ad profile"
        else:
            reason = "content"
        return Decision(
            event=Event.NO_CHANGE,
            timestamp=timestamp,
            ad_profile=profile,
            confidence=self._confidence(metrics),
            reason=reason,
            metrics=metrics,
        )

    def _update_in_ad(
        self,
        features: Features,
        timestamp: float,
        profile: bool,
        metrics: dict[str, float],
    ) -> Decision:
        cfg = self.config
        started = self._ad_started_at if self._ad_started_at is not None else timestamp
        elapsed = timestamp - started
        metrics["ad_elapsed_seconds"] = elapsed

        if elapsed >= cfg.max_ad_seconds:
            self.reject()
            return Decision(
                event=Event.AD_ENDED,
                timestamp=timestamp,
                ad_profile=False,
                confidence=0.0,
                reason=(
                    f"failsafe: {elapsed:.0f}s exceeds max_ad_seconds="
                    f"{cfg.max_ad_seconds:.0f}s"
                ),
                metrics=metrics,
            )

        if profile:
            self._non_ad_streak = 0
            return Decision(
                event=Event.NO_CHANGE,
                timestamp=timestamp,
                ad_profile=True,
                confidence=self._confidence(metrics),
                reason=f"ad continues ({elapsed:.0f}s)",
                metrics=metrics,
            )

        self._non_ad_streak += 1
        if (
            self._non_ad_streak >= cfg.ad_end_windows
            and elapsed >= cfg.min_ad_seconds
        ):
            self.reject()
            # The first content windows re-seed the baseline immediately, so a
            # long ad block does not leave us comparing against stale numbers.
            if not features.is_silence:
                self.baseline.update(features, cfg.baseline_alpha)
            return Decision(
                event=Event.AD_ENDED,
                timestamp=timestamp,
                ad_profile=False,
                confidence=0.0,
                reason=(
                    f"{self._non_ad_streak} non-ad windows after {elapsed:.0f}s"
                ),
                metrics=metrics,
            )

        return Decision(
            event=Event.NO_CHANGE,
            timestamp=timestamp,
            ad_profile=False,
            confidence=self._confidence(metrics),
            reason=(
                f"ad profile lapsed ({self._non_ad_streak}/{cfg.ad_end_windows}"
                f" windows, {elapsed:.0f}s elapsed)"
            ),
            metrics=metrics,
        )
