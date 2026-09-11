"""A detector that treats a seam as a door and loudness as the sign on it.

The legacy detector asks "does this window sound like an ad?" every second, and
its answer flickers because the question has no stable answer on modern TV
audio. This asks a different question, far less often: at the moment two pieces
of audio were joined, did the level step up into advertising, or back down into
the programme?

    enter   a seam where the audio steps UP by door_enter_step_db
    stay    seams inside the break are loud-to-loud and change nothing
    leave   a seam where the level returns to within door_exit_margin_db of
            the show level measured before the break began

Between seams it makes no judgement at all, which is the point: an ad is not
re-decided sixty times a minute, so the mute cannot flicker.

What the measurement says (scripts/seam_report.py, 25 breaks over 287 min):
seams are a reliable door -- 23 of 25 break starts and 21 of 25 break ends have
one within 2s -- but the loudness step is a weak sign. Break starts step a
median +1.7 dB against -0.3 dB for content seams, and content seams outnumber
them roughly 50 to 1. Exit looks better than entry: audio after an inside-break
seam sits about +4.4 dB above show level while after a break-end seam it sits
-1.1 dB, so the sign flips where it should.

Voters are deliberately ignored here. This is a test of one idea on its own.
"""

from __future__ import annotations

import logging
import statistics
from collections import deque

from .config import DetectionConfig
from .detector import Decision, Event, SeamTracker
from .features import Features

log = logging.getLogger(__name__)


def _clamp01(value: float) -> float:
    return max(0.0, min(1.0, value))


class SeamDoorDetector:
    """Detector protocol: update / reset / reject, returning Decision.

    Holds no clock. Every timestamp arrives as an argument, so offline replay
    and the live service see identical behaviour.
    """

    def __init__(self, config: DetectionConfig, window_seconds: float = 1.0) -> None:
        self.config = config
        self.window_seconds = max(1e-6, window_seconds)
        self._seams = SeamTracker(config.min_gap_seconds, config.max_gap_seconds)

        def windows_for(seconds: float) -> int:
            return max(1, int(round(seconds / self.window_seconds)))

        self._before: deque[float] = deque(maxlen=windows_for(config.door_before_seconds))
        self._fallback: deque[float] = deque(
            maxlen=windows_for(config.door_fallback_seconds)
        )
        self._after_target = windows_for(config.door_after_seconds)

        self._in_ad = False
        self._ad_started_at: float | None = None
        self._content_level: float | None = None
        # A seam is judged on the audio that follows it, which has not arrived
        # yet. Until it does, this holds the level from before the seam.
        self._pending_before: float | None = None
        self._pending_after: list[float] = []
        self._pending_gap = 0.0
        self._confidence = 0.0

    # ------------------------------------------------------------------ #
    # Detector protocol
    # ------------------------------------------------------------------ #

    def reset(self) -> None:
        """Stream restart or app switch: nothing learned still applies."""
        self._seams.reset()
        self._before.clear()
        self._fallback.clear()
        self.reject()

    def reject(self) -> None:
        """Drop ad state and any half-finished seam, keep the before-buffer.

        The buffer is a rolling picture of recent show audio. A rejected ad does
        not make the last ten seconds of television untrue, and throwing it away
        would leave the next seam with nothing to compare against.
        """
        self._in_ad = False
        self._ad_started_at = None
        self._pending_before = None
        self._pending_after = []
        self._pending_gap = 0.0

    @property
    def in_ad(self) -> bool:
        return self._in_ad

    def update(self, features: Features, timestamp: float) -> Decision:
        cfg = self.config
        gap = self._seams.gap(features)
        is_seam = self._seams.qualifies(gap)

        if is_seam:
            if self._pending_before is None:
                # Snapshot BEFORE this window is added: the window that closes
                # the gap is the first piece of "after" audio, not "before".
                self._pending_before = (
                    statistics.median(self._before) if self._before else None
                )
            # A second seam mid-collection restarts the collection but keeps the
            # original before_level: the sliver of audio between two seams is
            # not representative show audio.
            self._pending_after = []
            self._pending_gap = gap

        if not features.is_silence:
            if self._pending_before is not None:
                self._pending_after.append(features.rms_dbfs)
            self._before.append(features.rms_dbfs)
            self._fallback.append(features.rms_dbfs)

        metrics = self._metrics(features, gap)

        if self._in_ad:
            return self._update_in_ad(features, timestamp, metrics)
        return self._update_in_content(features, timestamp, metrics)

    # ------------------------------------------------------------------ #
    # Internals
    # ------------------------------------------------------------------ #

    def _collection_ready(self) -> bool:
        return (
            self._pending_before is not None
            and len(self._pending_after) >= self._after_target
        )

    def _take_collection(self) -> tuple[float, float, float]:
        before = float(self._pending_before)
        after = statistics.median(self._pending_after)
        gap = self._pending_gap
        self._pending_before = None
        self._pending_after = []
        self._pending_gap = 0.0
        return before, after, gap

    def _metrics(self, features: Features, gap: float) -> dict[str, float]:
        return {
            "rms_dbfs": features.rms_dbfs,
            "gap_seconds": gap,
            "collection_pending": float(self._pending_before is not None),
            "before_level": (
                self._pending_before if self._pending_before is not None else float("nan")
            ),
            "content_level": (
                self._content_level if self._content_level is not None else float("nan")
            ),
            "after_level": float("nan"),
            "step": float("nan"),
        }

    def _update_in_content(
        self, features: Features, timestamp: float, metrics: dict[str, float]
    ) -> Decision:
        cfg = self.config
        if not self._collection_ready():
            return Decision(
                event=Event.NO_CHANGE,
                timestamp=timestamp,
                ad_profile=False,
                confidence=0.0,
                reason=(
                    "listening after a seam"
                    if self._pending_before is not None else "content"
                ),
                metrics=metrics,
            )

        before, after, gap = self._take_collection()
        step = after - before
        metrics.update(after_level=after, before_level=before, step=step,
                       collection_pending=0.0)

        if step >= cfg.door_enter_step_db:
            self._in_ad = True
            self._ad_started_at = timestamp
            self._content_level = before
            # The fallback asks "have the last N seconds been at show level?".
            # At this instant the buffer is still full of the show audio that
            # came BEFORE the break, which answers yes and exits immediately.
            # It may only consider audio heard since the ad began.
            self._fallback.clear()
            self._confidence = _clamp01((step - cfg.door_enter_step_db) / 6.0)
            metrics["content_level"] = before
            return Decision(
                event=Event.AD_STARTED,
                timestamp=timestamp,
                ad_profile=True,
                confidence=self._confidence,
                reason=f"seam, {step:+.1f} dB vs show -> ad",
                metrics=metrics,
            )
        return Decision(
            event=Event.NO_CHANGE,
            timestamp=timestamp,
            ad_profile=False,
            confidence=0.0,
            reason=f"seam, {step:+.1f} dB -> not loud enough",
            metrics=metrics,
        )

    def _update_in_ad(
        self, features: Features, timestamp: float, metrics: dict[str, float]
    ) -> Decision:
        cfg = self.config
        started = self._ad_started_at if self._ad_started_at is not None else timestamp
        elapsed = timestamp - started
        metrics["ad_elapsed_seconds"] = elapsed
        exit_level = (self._content_level or 0.0) + cfg.door_exit_margin_db

        if elapsed >= cfg.max_ad_seconds:
            self.reject()
            return Decision(
                event=Event.AD_ENDED, timestamp=timestamp, ad_profile=False,
                confidence=0.0,
                reason=f"failsafe: {elapsed:.0f}s over max_ad_seconds",
                metrics=metrics,
            )

        if self._collection_ready():
            before, after, gap = self._take_collection()
            metrics.update(after_level=after, step=after - before,
                           collection_pending=0.0)
            if after <= exit_level and elapsed >= cfg.min_ad_seconds:
                self.reject()
                return Decision(
                    event=Event.AD_ENDED, timestamp=timestamp, ad_profile=False,
                    confidence=0.0,
                    reason=f"seam, back to show level ({after:.1f} dBFS)",
                    metrics=metrics,
                )
            return Decision(
                event=Event.NO_CHANGE, timestamp=timestamp, ad_profile=True,
                confidence=self._confidence,
                reason=f"seam inside break, still loud ({after:.1f} dBFS)",
                metrics=metrics,
            )

        # No seam. A missed exit seam would otherwise hold the mute to the
        # failsafe, so a sustained return to show level ends it too.
        if (
            elapsed >= cfg.min_ad_seconds
            and len(self._fallback) >= self._fallback.maxlen
            and statistics.median(self._fallback) <= exit_level
        ):
            self.reject()
            return Decision(
                event=Event.AD_ENDED, timestamp=timestamp, ad_profile=False,
                confidence=0.0,
                reason=f"fallback: quiet for {cfg.door_fallback_seconds:.0f}s",
                metrics=metrics,
            )

        return Decision(
            event=Event.NO_CHANGE, timestamp=timestamp, ad_profile=True,
            confidence=self._confidence,
            reason=f"ad continues ({elapsed:.0f}s)",
            metrics=metrics,
        )


def build_detector(config: DetectionConfig, window_seconds: float, **voters):
    """Return whichever detector detection.mode asks for.

    ``legacy`` gets the voters; ``seam_door`` never does -- it is a test of one
    idea, and handing it an ensemble would make the result unattributable.
    """
    from .detector import HeuristicDetector

    if getattr(config, "mode", "legacy") == "seam_door":
        log.info("detection mode: seam_door (voters ignored)")
        return SeamDoorDetector(config, window_seconds)
    return HeuristicDetector(config, window_seconds, **voters)
