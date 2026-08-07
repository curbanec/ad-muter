"""State machine and main loop.

    CONTENT ──AD_STARTED──> AD_SUSPECTED ──confirmed──> MUTED
       ^                        │                         │
       └────confirmation────────┘                         │
            failed                                        │
       ^                                                  │
       └──────────AD_ENDED / failsafe / disarmed──────────┘

The confirmation window is the whole point of AD_SUSPECTED: the detector fires
on the first ad-looking window, and we insist the profile survives another
window or two before touching the TV. A loud, compressed action scene right
after a quiet one can produce a single convincing window; it rarely produces
three in a row directly after a silent seam.

Two independent failsafes make sure the TV can never stay muted forever: the
detector's ``max_ad_seconds`` and the controller's ``max_mute_seconds``.
"""

from __future__ import annotations

import logging
from enum import Enum
from typing import Iterable, Protocol

from .capture import AudioWindow
from .config import Config
from .detector import Decision, Detector, Event
from .features import Features, compute_features
from .logging_setup import FeatureLogger
from .roku import RokuClient

log = logging.getLogger(__name__)


class State(str, Enum):
    CONTENT = "CONTENT"
    AD_SUSPECTED = "AD_SUSPECTED"
    MUTED = "MUTED"


class WindowSource(Protocol):
    """Anything that produces AudioWindows (live capture, or a WAV replay)."""

    def windows(self) -> Iterable[AudioWindow]: ...

    def stop(self) -> None: ...


class Controller:
    """Owns the state machine and drives capture -> features -> detector -> TV."""

    def __init__(
        self,
        capture: WindowSource,
        detector: Detector,
        roku: RokuClient,
        config: Config,
        feature_logger: FeatureLogger | None = None,
    ) -> None:
        self.capture = capture
        self.detector = detector
        self.roku = roku
        self.config = config
        self.feature_logger = feature_logger

        self.state = State.CONTENT
        self._stop = False
        self._pending = 0
        self._suspect_reason = ""
        self._muted_at: float | None = None
        self._we_muted = False
        self._unmute_pending: str | None = None
        self._armed = not config.roku.netflix_only
        self._armed_checked_at: float | None = None
        self._last_reason = ""
        self._windows_seen = 0

    # ------------------------------------------------------------------ #
    # Main loop
    # ------------------------------------------------------------------ #

    def run(self) -> None:
        log.info(
            "admuter starting: %s -> %s (netflix_only=%s)",
            self.config.audio.device,
            self.config.roku.base_url,
            self.config.roku.netflix_only,
        )
        try:
            for window in self.capture.windows():
                if self._stop:
                    break
                self.process_window(window)
        finally:
            self.shutdown()

    def request_stop(self) -> None:
        """Signal-handler entry point: finish the current window and exit."""
        if not self._stop:
            log.info("stop requested")
        self._stop = True
        try:
            self.capture.stop()
        except Exception:  # pragma: no cover - defensive
            log.debug("capture.stop() raised", exc_info=True)

    def shutdown(self) -> None:
        """Leave the TV audible. Called on every exit path, including crashes."""
        try:
            if self._we_muted and self.roku.is_muted:
                if self.roku.unmute():
                    log.info("UNMUTE — shutdown")
                else:
                    log.error(
                        "could not unmute during shutdown — the TV may be left "
                        "muted; press Mute on the remote"
                    )
        finally:
            if self.feature_logger is not None:
                self.feature_logger.close()

    # ------------------------------------------------------------------ #
    # Per-window processing
    # ------------------------------------------------------------------ #

    def process_window(self, window: AudioWindow) -> Decision | None:
        """Handle one window. Returns the detector decision, if we were armed."""
        self._windows_seen += 1
        features = compute_features(
            window.samples,
            window.sample_rate,
            silence_dbfs=self.config.detection.silence_dbfs,
            frame_seconds=self.config.audio.frame_seconds,
        )

        if window.stream_restarted and window.index > 0:
            log.info("capture restarted — resetting detector state")
            self._reset_after_break("capture restarted")

        decision: Decision | None = None
        if self._check_armed(window.timestamp):
            decision = self.detector.update(features, window.timestamp)
            self._advance(decision, window.timestamp)
        else:
            self._disarm()

        self._log_window(window, features, decision)
        return decision

    def _advance(self, decision: Decision, timestamp: float) -> None:
        if self.state is State.MUTED:
            self._advance_muted(decision, timestamp)
        elif self.state is State.AD_SUSPECTED:
            self._advance_suspected(decision, timestamp)
        else:
            self._advance_content(decision, timestamp)

    def _advance_content(self, decision: Decision, timestamp: float) -> None:
        if decision.event is not Event.AD_STARTED:
            return
        self.state = State.AD_SUSPECTED
        self._pending = 1
        self._suspect_reason = decision.reason
        log.info(
            "AD_SUSPECTED (1/%d) conf=%.2f — %s",
            self.config.controller.confirm_windows,
            decision.confidence,
            decision.reason,
        )
        if self._pending >= self.config.controller.confirm_windows:
            self._mute(decision.reason, timestamp)

    def _advance_suspected(self, decision: Decision, timestamp: float) -> None:
        confirm = self.config.controller.confirm_windows
        if decision.event is Event.AD_ENDED:
            log.info("ad ended before confirmation — %s", decision.reason)
            self._to_content()
            return
        if not decision.ad_profile:
            # Prefer a missed ad over muting real content.
            log.info(
                "confirmation failed at %d/%d windows — staying unmuted (%s)",
                self._pending,
                confirm,
                decision.reason,
            )
            self._to_content()
            self.detector.reject()
            return

        self._pending += 1
        if self._pending >= confirm:
            self._mute(self._suspect_reason or decision.reason, timestamp)
        else:
            log.info(
                "AD_SUSPECTED (%d/%d) conf=%.2f", self._pending, confirm, decision.confidence
            )

    def _advance_muted(self, decision: Decision, timestamp: float) -> None:
        if self._unmute_pending is not None:
            self._unmute(self._unmute_pending)
            if self.state is State.CONTENT:
                return

        if decision.event is Event.AD_ENDED:
            self._unmute(decision.reason)
            return

        elapsed = timestamp - (self._muted_at if self._muted_at is not None else timestamp)
        if elapsed >= self.config.controller.max_mute_seconds:
            log.warning(
                "mute failsafe tripped after %.0fs — unmuting and resetting", elapsed
            )
            self.detector.reject()
            self._unmute(f"failsafe after {elapsed:.0f}s muted")

    # ------------------------------------------------------------------ #
    # TV actions
    # ------------------------------------------------------------------ #

    def _mute(self, reason: str, timestamp: float) -> None:
        if self.roku.mute():
            self.state = State.MUTED
            self._muted_at = timestamp
            self._we_muted = True
            self._unmute_pending = None
            log.info("MUTE (%d/%d windows) — %s", self._pending,
                     self.config.controller.confirm_windows, reason)
        else:
            log.warning("mute command failed — retrying on next window")

    def _unmute(self, reason: str) -> None:
        if self.roku.unmute():
            log.info("UNMUTE — %s", reason)
            self._unmute_pending = None
            self._to_content()
        else:
            self._unmute_pending = reason
            log.warning("unmute command failed — retrying on next window")

    def _to_content(self) -> None:
        self.state = State.CONTENT
        self._pending = 0
        self._suspect_reason = ""
        self._muted_at = None

    def _reset_after_break(self, reason: str) -> None:
        """Recover from a discontinuity: unmute if needed, forget history."""
        self.detector.reset()
        if self.state is State.MUTED or self.roku.is_muted:
            self._unmute(reason)
        else:
            self._to_content()

    # ------------------------------------------------------------------ #
    # Netflix gating
    # ------------------------------------------------------------------ #

    def _check_armed(self, timestamp: float) -> bool:
        if not self.config.roku.netflix_only:
            return True
        interval = self.config.controller.app_check_seconds
        due = (
            self._armed_checked_at is None
            or timestamp - self._armed_checked_at >= interval
        )
        if due:
            self._armed_checked_at = timestamp
            active = self.roku.is_netflix_active()
            if active is None:
                log.debug("could not query active app — keeping armed=%s", self._armed)
            elif active != self._armed:
                log.info(
                    "Netflix %s — detection %s",
                    "active" if active else "not active",
                    "armed" if active else "disarmed",
                )
                self._armed = active
                if active:
                    self.detector.reset()
        return self._armed

    def _disarm(self) -> None:
        if self.state is not State.CONTENT or self.roku.is_muted:
            self._reset_after_break("Netflix no longer active")

    # ------------------------------------------------------------------ #
    # Logging
    # ------------------------------------------------------------------ #

    def _log_window(
        self, window: AudioWindow, features: Features, decision: Decision | None
    ) -> None:
        log.debug(
            "win %d t=%.1f rms=%.1f dBFS peak=%.1f crest=%.1f dB centroid=%.0f Hz "
            "silence=%.0f%% gap=%.2fs state=%s",
            window.index,
            window.timestamp,
            features.rms_dbfs,
            features.peak_dbfs,
            features.crest_db,
            features.spectral_centroid_hz,
            features.silence_ratio * 100.0,
            features.max_silence_run_seconds,
            self.state.value,
        )
        if decision is not None:
            self._log_decision(decision)
        if self.feature_logger is not None:
            self.feature_logger.log(
                index=window.index,
                timestamp=window.timestamp,
                features=features,
                decision=decision,
                state=self.state.value,
                muted=self.roku.is_muted,
            )

    def _log_decision(self, decision: Decision) -> None:
        """INFO for anything that changed; steady state is deduplicated.

        A Pi writing one INFO line per second forever is a journal problem, so
        repeated identical NO_CHANGE reasons drop to DEBUG with a periodic
        heartbeat. Set logging.verbose_decisions to log literally every window.
        """
        cfg = self.config.logging
        changed = decision.is_change or decision.reason != self._last_reason
        heartbeat = (
            cfg.decision_heartbeat_windows > 0
            and self._windows_seen % cfg.decision_heartbeat_windows == 0
        )
        self._last_reason = decision.reason
        message = "%s state=%s ad_profile=%s conf=%.2f — %s"
        args = (
            decision.event.value,
            self.state.value,
            decision.ad_profile,
            decision.confidence,
            decision.reason,
        )
        if cfg.verbose_decisions or changed or heartbeat:
            log.info(message, *args)
        else:
            log.debug(message, *args)
