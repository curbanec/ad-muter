"""Feature history -> transition events.

Phase 1 is deliberately rule-based. The interface (`Detector`) is the seam a
Phase 2 ML classifier drops into: as long as something accepts a `Features` plus
a timestamp and returns a `Decision`, the controller does not care how the call
is implemented.

Bias throughout: when the evidence is thin, emit NO_CHANGE. A missed ad is
annoying; muting real dialogue is worse.
"""

from __future__ import annotations

import math
import statistics
from collections import deque
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

    Loudness can be judged two ways. By default it is the delta over the
    baseline described above. Setting ``ad_absolute_dbfs`` switches it to a
    fixed dBFS bar over a median of the last ``loudness_median_windows``
    windows; the hysteresis pair becomes ``ad_absolute_dbfs`` /
    ``ad_stay_absolute_dbfs`` and everything else is unchanged. The delta
    measure decays to nothing on long breaks -- the baseline is an EMA of
    recent audio, so it climbs to meet the ad it is supposed to be contrasted
    against -- while an absolute bar does not move.
    """

    def __init__(
        self,
        config: DetectionConfig,
        window_seconds: float = 1.0,
        ml_voter: object | None = None,
    ) -> None:
        self.config = config
        # A second opinion on the inner per-window question only. There is one
        # state machine; the voter never sees or touches it.
        self._ml_voter = ml_voter
        self.window_seconds = window_seconds
        self.baseline = Baseline()
        self._in_ad = False
        self._ad_started_at: float | None = None
        self._carry_silence = 0.0
        self._cue_at: float | None = None
        self._cue_gap = 0.0
        self._last_voiced: Features | None = None
        self._non_ad_streak = 0
        self._loudness: deque[float] = deque(maxlen=config.loudness_median_windows)

    # ------------------------------------------------------------------ #
    # Detector protocol
    # ------------------------------------------------------------------ #

    def reset(self) -> None:
        """Full reset, including the learned baseline."""
        self.baseline = Baseline()
        self.reject()
        self._carry_silence = 0.0
        self._last_voiced = None
        self._loudness.clear()

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

        self._loudness.append(features.rms_dbfs)
        smoothed_rms = statistics.median(self._loudness)

        # Hysteresis: the bar loudness must clear depends on state, and which
        # quantity is being judged depends on the mode.
        absolute = not math.isnan(cfg.ad_absolute_dbfs)
        if absolute:
            loudness_threshold = (
                cfg.ad_stay_absolute_dbfs if self._in_ad else cfg.ad_absolute_dbfs
            )
        else:
            loudness_threshold = (
                cfg.ad_stay_loudness_delta_db
                if self._in_ad
                else cfg.ad_loudness_delta_db
            )
        profile, profile_metrics = self._ad_profile(
            features, loudness_threshold, smoothed_rms
        )
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
            "absolute_mode": float(absolute),
            "loudness_threshold_db": loudness_threshold,
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
        self,
        features: Features,
        loudness_threshold_db: float,
        smoothed_rms_dbfs: float,
    ) -> tuple[bool, dict[str, float]]:
        """Does this window look like ad audio?

        ``loudness_threshold_db`` is the bar loudness must clear, already
        resolved by the caller for the current mode and ad state.

        In absolute mode the loudness test needs no learned baseline, so the
        profile can fire in the first seconds of a recording -- which is where a
        break that starts at 00:00 lives, and the delta mode could never see it.
        The crest test still wants a baseline; until there is one it is scored
        as neutral rather than left to veto every window.
        """
        cfg = self.config
        absolute = not math.isnan(cfg.ad_absolute_dbfs)
        ready = self.baseline.ready(cfg.baseline_min_windows)
        metrics = {
            "loudness_delta_db": 0.0,
            "crest_delta_db": 0.0,
            "smoothed_rms_dbfs": smoothed_rms_dbfs,
        }
        if features.is_silence or (not absolute and not ready):
            return self._second_opinion(False, features, metrics)

        if ready:
            metrics["loudness_delta_db"] = features.rms_dbfs - float(
                self.baseline.rms_dbfs
            )
            metrics["crest_delta_db"] = float(self.baseline.crest_db) - features.crest_db

        if absolute:
            louder = smoothed_rms_dbfs >= loudness_threshold_db
        else:
            louder = metrics["loudness_delta_db"] >= loudness_threshold_db
        squashed = metrics["crest_delta_db"] >= cfg.ad_crest_delta_db
        return self._second_opinion(louder and squashed, features, metrics)

    def _second_opinion(
        self, heuristic: bool, features: Features, metrics: dict[str, float]
    ) -> tuple[bool, dict[str, float]]:
        """Fold the model's per-window vote into the heuristic's, if there is one.

        The combination mirrors the loudness hysteresis directly above it:

            entering an ad   heuristic AND ml   -- both must agree to start
            staying in an ad heuristic OR  ml   -- either can hold it open

        That asymmetry is the point. Muting real dialogue is the failure that
        ruins the experience, so starting a mute needs two votes; unmuting early in
        the middle of a break is merely annoying, so one voter can keep it
        alive. The same reasoning already sets ad_stay_loudness_delta_db below
        ad_loudness_delta_db.

        With no voter configured this returns the heuristic untouched and adds
        no keys, so an install that has never heard of Phase 2 logs exactly
        what it logged before.
        """
        if self._ml_voter is None:
            return heuristic, metrics

        ml_says, probability = self._ml_voter.says_ad(features, self.baseline)
        metrics["heuristic_says_ad"] = float(heuristic)
        metrics["ml_says_ad"] = float(ml_says)
        metrics["ml_probability"] = probability
        metrics["voters_disagree"] = float(ml_says != heuristic)

        if not self.config.ml_vote_enabled:
            # Shadow mode: recorded, never counted.
            return heuristic, metrics
        if self._in_ad:
            return heuristic or ml_says, metrics
        return heuristic and ml_says, metrics

    def _loudness_reason(self, metrics: dict[str, float]) -> str:
        """Phrase the loudness evidence in the units the decision was made in."""
        if not math.isnan(self.config.ad_absolute_dbfs):
            return (
                f"loudness {metrics['smoothed_rms_dbfs']:.1f}dBFS "
                f"(bar {metrics['loudness_threshold_db']:.1f})"
            )
        return f"loudness +{metrics['loudness_delta_db']:.1f}dB vs baseline"

    def _confidence(self, metrics: dict[str, float]) -> float:
        """Rough 0-1 score, for logging only — nothing decides on it.

        Loudness is normalised against whichever bar actually applied to this
        window, so a window that clears only the lower ``stay`` bar mid-ad is
        not reported as barely-confident against the higher ``enter`` bar it
        was never judged by.
        """
        cfg = self.config
        if not math.isnan(cfg.ad_absolute_dbfs):
            # Absolute mode: dB past the bar, scaled over a 12 dB span. A ratio
            # against the threshold itself would be nonsense -- it is negative.
            threshold = metrics.get("loudness_threshold_db", cfg.ad_absolute_dbfs)
            over = metrics.get("smoothed_rms_dbfs", threshold) - threshold
            loud = _clamp01(over / 12.0)
        else:
            threshold = metrics.get("loudness_threshold_db", cfg.ad_loudness_delta_db)
            loud = _clamp01(
                metrics.get("loudness_delta_db", 0.0) / max(threshold, 1e-6)
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
                    f"{self._loudness_reason(metrics)} / "
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
