"""Controller state-machine tests — scripted detector, fake TV, no hardware."""

from __future__ import annotations

import dataclasses

import numpy as np
import pytest

from admuter.capture import AudioWindow
from admuter.config import Config
from admuter.controller import Controller, State
from admuter.detector import Decision, Event
from admuter.features import Features

SR = 48000


class ScriptedDetector:
    """Replays a list of (event, ad_profile) pairs, holding the last one."""

    def __init__(self, script: list[tuple[Event, bool]]) -> None:
        self.script = list(script)
        self.calls = 0
        self.resets = 0
        self.rejects = 0

    def update(self, features: Features, timestamp: float) -> Decision:
        index = min(self.calls, len(self.script) - 1)
        event, ad_profile = self.script[index]
        self.calls += 1
        return Decision(
            event=event,
            timestamp=timestamp,
            ad_profile=ad_profile,
            confidence=0.9 if ad_profile else 0.0,
            reason="scripted",
            metrics={},
        )

    def reset(self) -> None:
        self.resets += 1

    def reject(self) -> None:
        self.rejects += 1


class FakeRoku:
    """Same surface as RokuClient, with failure injection."""

    def __init__(self, netflix: bool | None = True, fail: bool = False) -> None:
        self.is_muted = False
        self.netflix = netflix
        self.fail = fail
        self.toggles = 0
        self.app_queries = 0

    def mute(self) -> bool:
        if self.is_muted:
            return True
        if self.fail:
            return False
        self.is_muted = True
        self.toggles += 1
        return True

    def unmute(self) -> bool:
        if not self.is_muted:
            return True
        if self.fail:
            return False
        self.is_muted = False
        self.toggles += 1
        return True

    def is_netflix_active(self) -> bool | None:
        self.app_queries += 1
        return self.netflix


class FakeCapture:
    def __init__(self, windows: list[AudioWindow]) -> None:
        self._windows = windows
        self.stopped = False

    def windows(self):
        return iter(self._windows)

    def stop(self) -> None:
        self.stopped = True


def make_config(**overrides) -> Config:
    data = {
        "roku": {"netflix_only": False},
        "controller": {"confirm_windows": 2, "max_mute_seconds": 130.0},
    }
    for section, values in overrides.items():
        data.setdefault(section, {}).update(values)
    return Config.from_dict(data)


def window(index: int, restarted: bool = False, sr: int = SR) -> AudioWindow:
    return AudioWindow(
        samples=np.zeros(sr // 8, dtype=np.float32),
        sample_rate=sr,
        index=index,
        timestamp=float(index),
        stream_restarted=restarted,
    )


def build(script, config=None, roku=None):
    config = config or make_config()
    roku = roku or FakeRoku()
    detector = ScriptedDetector(script)
    controller = Controller(FakeCapture([]), detector, roku, config)
    return controller, detector, roku


def drive(controller: Controller, count: int, start: int = 0) -> None:
    for i in range(start, start + count):
        controller.process_window(window(i))


# --------------------------------------------------------------------------- #
# Muting
# --------------------------------------------------------------------------- #


def test_mute_requires_the_confirmation_window():
    controller, _, roku = build([(Event.AD_STARTED, True), (Event.NO_CHANGE, True)])
    controller.process_window(window(0))
    assert controller.state is State.AD_SUSPECTED
    assert roku.is_muted is False  # one convincing window is not enough

    controller.process_window(window(1))
    assert controller.state is State.MUTED
    assert roku.is_muted is True


def test_confirm_windows_of_one_mutes_immediately():
    config = make_config(controller={"confirm_windows": 1})
    controller, _, roku = build([(Event.AD_STARTED, True)], config=config)
    controller.process_window(window(0))
    assert controller.state is State.MUTED
    assert roku.is_muted is True


def test_three_window_confirmation():
    config = make_config(controller={"confirm_windows": 3})
    controller, _, roku = build(
        [(Event.AD_STARTED, True), (Event.NO_CHANGE, True)], config=config
    )
    drive(controller, 2)
    assert roku.is_muted is False
    controller.process_window(window(2))
    assert roku.is_muted is True


def test_failed_confirmation_leaves_the_tv_alone():
    """The whole point of AD_SUSPECTED: a single loud window must not mute."""
    controller, detector, roku = build(
        [(Event.AD_STARTED, True), (Event.NO_CHANGE, False)]
    )
    drive(controller, 4)
    assert controller.state is State.CONTENT
    assert roku.is_muted is False
    assert roku.toggles == 0
    assert detector.rejects == 1


def test_ad_ended_during_confirmation_returns_to_content():
    controller, _, roku = build([(Event.AD_STARTED, True), (Event.AD_ENDED, False)])
    drive(controller, 2)
    assert controller.state is State.CONTENT
    assert roku.is_muted is False


def test_mute_failure_is_retried_on_the_next_window():
    roku = FakeRoku(fail=True)
    controller, _, _ = build(
        [(Event.AD_STARTED, True), (Event.NO_CHANGE, True)], roku=roku
    )
    drive(controller, 2)
    assert controller.state is State.AD_SUSPECTED
    assert roku.is_muted is False

    roku.fail = False
    controller.process_window(window(2))
    assert controller.state is State.MUTED
    assert roku.is_muted is True


# --------------------------------------------------------------------------- #
# Unmuting
# --------------------------------------------------------------------------- #


def test_ad_ended_unmutes():
    controller, _, roku = build(
        [
            (Event.AD_STARTED, True),
            (Event.NO_CHANGE, True),
            (Event.NO_CHANGE, True),
            (Event.AD_ENDED, False),
        ]
    )
    drive(controller, 4)
    assert controller.state is State.CONTENT
    assert roku.is_muted is False
    assert roku.toggles == 2


def test_mute_failsafe_unmutes_without_an_ad_ended_event():
    """A detector that never says the ad ended must not leave the TV muted."""
    config = make_config(
        controller={"confirm_windows": 1, "max_mute_seconds": 130.0},
        detection={"max_ad_seconds": 120.0},
    )
    controller, detector, roku = build(
        [(Event.AD_STARTED, True), (Event.NO_CHANGE, True)], config=config
    )
    controller.process_window(window(0))
    assert roku.is_muted is True
    drive(controller, 129, start=1)
    assert roku.is_muted is True  # not yet
    controller.process_window(window(130))
    assert roku.is_muted is False
    assert controller.state is State.CONTENT
    assert detector.rejects == 1


def test_unmute_failure_is_retried_on_the_next_window():
    roku = FakeRoku()
    config = make_config(controller={"confirm_windows": 1})
    controller, _, _ = build(
        [(Event.AD_STARTED, True), (Event.AD_ENDED, False), (Event.NO_CHANGE, False)],
        config=config,
        roku=roku,
    )
    controller.process_window(window(0))
    assert roku.is_muted is True

    roku.fail = True
    controller.process_window(window(1))  # AD_ENDED, but the command fails
    assert controller.state is State.MUTED
    assert roku.is_muted is True

    roku.fail = False
    controller.process_window(window(2))  # NO_CHANGE, but the retry goes through
    assert controller.state is State.CONTENT
    assert roku.is_muted is False


# --------------------------------------------------------------------------- #
# Gating and discontinuities
# --------------------------------------------------------------------------- #


def test_detection_is_disarmed_when_netflix_is_not_active():
    config = make_config(roku={"netflix_only": True}, controller={"app_check_seconds": 0.0})
    roku = FakeRoku(netflix=False)
    controller, detector, _ = build(
        [(Event.AD_STARTED, True)], config=config, roku=roku
    )
    drive(controller, 5)
    assert detector.calls == 0
    assert roku.is_muted is False


def test_arming_when_netflix_appears_resets_the_detector():
    config = make_config(roku={"netflix_only": True}, controller={"app_check_seconds": 0.0})
    roku = FakeRoku(netflix=False)
    controller, detector, _ = build(
        [(Event.NO_CHANGE, False)], config=config, roku=roku
    )
    controller.process_window(window(0))
    assert detector.calls == 0

    roku.netflix = True
    controller.process_window(window(1))
    assert detector.resets == 1
    assert detector.calls == 1


def test_leaving_netflix_while_muted_unmutes():
    config = make_config(
        roku={"netflix_only": True},
        controller={"confirm_windows": 1, "app_check_seconds": 0.0},
    )
    roku = FakeRoku(netflix=True)
    controller, _, _ = build([(Event.AD_STARTED, True)], config=config, roku=roku)
    controller.process_window(window(0))
    assert roku.is_muted is True

    roku.netflix = False
    controller.process_window(window(1))
    assert roku.is_muted is False
    assert controller.state is State.CONTENT


def test_unreachable_tv_keeps_the_previous_arm_state():
    """A failed active-app query must not silently disarm detection."""
    config = make_config(roku={"netflix_only": True}, controller={"app_check_seconds": 0.0})
    roku = FakeRoku(netflix=True)
    controller, detector, _ = build(
        [(Event.NO_CHANGE, False)], config=config, roku=roku
    )
    controller.process_window(window(0))
    assert detector.calls == 1

    roku.netflix = None  # query fails
    controller.process_window(window(1))
    assert detector.calls == 2


def test_capture_restart_resets_state_and_unmutes():
    config = make_config(controller={"confirm_windows": 1})
    controller, detector, roku = build(
        [(Event.AD_STARTED, True), (Event.NO_CHANGE, False)], config=config
    )
    controller.process_window(window(0))
    assert roku.is_muted is True

    controller.process_window(window(1, restarted=True))
    assert detector.resets == 1
    assert roku.is_muted is False
    assert controller.state is State.CONTENT


def test_first_window_of_the_stream_is_not_treated_as_a_restart():
    controller, detector, _ = build([(Event.NO_CHANGE, False)])
    controller.process_window(window(0, restarted=True))
    assert detector.resets == 0


# --------------------------------------------------------------------------- #
# Lifecycle
# --------------------------------------------------------------------------- #


def test_shutdown_unmutes_a_tv_we_muted():
    config = make_config(controller={"confirm_windows": 1})
    controller, _, roku = build([(Event.AD_STARTED, True)], config=config)
    controller.process_window(window(0))
    assert roku.is_muted is True
    controller.shutdown()
    assert roku.is_muted is False


def test_shutdown_leaves_a_tv_we_did_not_mute_alone():
    controller, _, roku = build([(Event.NO_CHANGE, False)])
    roku.is_muted = True  # the user muted it with the remote
    controller.process_window(window(0))
    controller.shutdown()
    assert roku.is_muted is True


def test_run_processes_every_window_then_shuts_down():
    config = make_config(controller={"confirm_windows": 1})
    detector = ScriptedDetector([(Event.NO_CHANGE, False)])
    roku = FakeRoku()
    capture = FakeCapture([window(i) for i in range(5)])
    controller = Controller(capture, detector, roku, config)
    controller.run()
    assert detector.calls == 5


def test_request_stop_halts_the_loop_and_stops_capture():
    config = make_config()
    detector = ScriptedDetector([(Event.NO_CHANGE, False)])
    roku = FakeRoku()
    capture = FakeCapture([window(i) for i in range(5)])
    controller = Controller(capture, detector, roku, config)
    controller.request_stop()
    controller.run()
    assert detector.calls == 0
    assert capture.stopped is True


def test_feature_logger_receives_a_row_per_window(tmp_path):
    from admuter.logging_setup import FeatureLogger

    path = tmp_path / "features.jsonl"
    config = make_config()
    detector = ScriptedDetector([(Event.NO_CHANGE, False)])
    controller = Controller(
        FakeCapture([]), detector, FakeRoku(), config, FeatureLogger(path, "jsonl")
    )
    drive(controller, 3)
    controller.shutdown()
    lines = path.read_text().strip().splitlines()
    assert len(lines) == 3
    assert '"state": "CONTENT"' in lines[0]


def test_csv_feature_log_has_a_single_header(tmp_path):
    from admuter.logging_setup import FeatureLogger

    path = tmp_path / "features.csv"
    config = make_config()
    controller = Controller(
        FakeCapture([]),
        ScriptedDetector([(Event.NO_CHANGE, False)]),
        FakeRoku(),
        config,
        FeatureLogger(path, "csv"),
    )
    drive(controller, 3)
    controller.shutdown()
    rows = path.read_text().strip().splitlines()
    assert len(rows) == 4
    assert rows[0].startswith("index,timestamp,state")


# --------------------------------------------------------------------------- #
# Config surface used by the controller
# --------------------------------------------------------------------------- #


def test_config_rejects_unknown_keys():
    from admuter.config import ConfigError

    with pytest.raises(ConfigError, match="unknown key"):
        Config.from_dict({"detection": {"ad_crest_delta": 2.0}})


def test_config_rejects_a_failsafe_that_can_never_fire():
    from admuter.config import ConfigError

    with pytest.raises(ConfigError, match="max_mute_seconds"):
        Config.from_dict(
            {
                "detection": {"max_ad_seconds": 200.0},
                "controller": {"max_mute_seconds": 100.0},
            }
        )


def test_config_coerces_integers_to_floats():
    config = Config.from_dict({"detection": {"max_ad_seconds": 90}})
    assert config.detection.max_ad_seconds == pytest.approx(90.0)
    assert isinstance(config.detection.max_ad_seconds, float)


def test_config_defaults_round_trip_through_dataclasses():
    config = Config.from_dict({})
    assert config.audio.frames_per_window == 48000
    assert config.roku.base_url == "http://192.168.0.12:8060"
    assert dataclasses.replace(config.roku, host="10.0.0.5").base_url.startswith(
        "http://10.0.0.5"
    )
