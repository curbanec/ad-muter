"""Roku ECP client tests — mocked HTTP, no live TV."""

from __future__ import annotations

import dataclasses

import pytest
import requests

from admuter.config import RokuConfig
from admuter.roku import ActiveApp, RokuClient

ACTIVE_APP_NETFLIX = (
    '<?xml version="1.0" encoding="UTF-8" ?>\n'
    '<active-app><app id="12" type="appl" version="8.1.4">Netflix</app></active-app>'
)
ACTIVE_APP_HOME = (
    '<?xml version="1.0" encoding="UTF-8" ?>\n'
    "<active-app><app>Roku</app></active-app>"
)
DEVICE_INFO = (
    '<?xml version="1.0" encoding="UTF-8" ?>\n'
    "<device-info><model-name>Hisense Roku TV</model-name>"
    "<serial-number>X0123456789</serial-number>"
    "<friendly-device-name>Living Room TV</friendly-device-name></device-info>"
)


class FakeResponse:
    def __init__(self, text: str = "", status: int = 200) -> None:
        self.text = text
        self.status_code = status

    def raise_for_status(self) -> None:
        if self.status_code >= 400:
            raise requests.HTTPError(f"{self.status_code} error")


class FakeSession:
    """Records every ECP call; can be told to fail."""

    def __init__(self, get_body: str = "", fail: bool = False) -> None:
        self.posts: list[str] = []
        self.gets: list[str] = []
        self.get_body = get_body
        self.fail = fail
        self.closed = False

    def post(self, url: str, data=None, timeout=None):
        self.posts.append(url)
        if self.fail:
            raise requests.ConnectionError("network is unreachable")
        return FakeResponse()

    def get(self, url: str, timeout=None):
        self.gets.append(url)
        if self.fail:
            raise requests.Timeout("timed out")
        return FakeResponse(self.get_body)

    def close(self) -> None:
        self.closed = True


@pytest.fixture
def config() -> RokuConfig:
    return RokuConfig(host="192.168.0.12", port=8060, timeout_seconds=2.0)


def client(config: RokuConfig, session: FakeSession, **kwargs) -> RokuClient:
    return RokuClient(config, session=session, **kwargs)


# --------------------------------------------------------------------------- #
# Mute state tracking — VolumeMute is a toggle
# --------------------------------------------------------------------------- #


def test_mute_sends_one_toggle(config):
    session = FakeSession()
    roku = client(config, session)
    assert roku.mute() is True
    assert roku.is_muted is True
    assert session.posts == ["http://192.168.0.12:8060/keypress/VolumeMute"]


def test_muting_twice_sends_only_one_toggle(config):
    session = FakeSession()
    roku = client(config, session)
    roku.mute()
    roku.mute()
    roku.mute()
    assert len(session.posts) == 1
    assert roku.is_muted is True


def test_unmute_when_already_unmuted_is_a_no_op(config):
    session = FakeSession()
    roku = client(config, session)
    assert roku.unmute() is True
    assert session.posts == []
    assert roku.is_muted is False


def test_mute_unmute_round_trip(config):
    session = FakeSession()
    roku = client(config, session)
    roku.mute()
    roku.unmute()
    assert len(session.posts) == 2
    assert roku.is_muted is False


def test_assume_muted_at_start_is_respected(config):
    session = FakeSession()
    roku = client(dataclasses.replace(config, assume_muted_at_start=True), session)
    assert roku.is_muted is True
    assert roku.mute() is True
    assert session.posts == []  # already muted: nothing sent
    roku.unmute()
    assert len(session.posts) == 1


def test_believed_state_is_unchanged_when_the_toggle_fails(config):
    session = FakeSession(fail=True)
    roku = client(config, session)
    assert roku.mute() is False
    assert roku.is_muted is False  # do not claim a mute we could not send


def test_set_believed_mute_state_resyncs_without_touching_the_tv(config):
    session = FakeSession()
    roku = client(config, session)
    roku.set_believed_mute_state(True)
    assert roku.is_muted is True
    assert session.posts == []


def test_dry_run_never_posts(config):
    session = FakeSession()
    roku = client(config, session, dry_run=True)
    assert roku.mute() is True
    assert roku.is_muted is True
    assert session.posts == []


# --------------------------------------------------------------------------- #
# Queries
# --------------------------------------------------------------------------- #


def test_active_app_parses_netflix(config):
    roku = client(config, FakeSession(ACTIVE_APP_NETFLIX))
    app = roku.active_app()
    assert app == ActiveApp(app_id="12", name="Netflix")
    assert roku.is_netflix_active() is True


def test_active_app_parses_home_screen(config):
    roku = client(config, FakeSession(ACTIVE_APP_HOME))
    assert roku.active_app() == ActiveApp(app_id="", name="Roku")
    assert roku.is_netflix_active() is False


def test_netflix_matches_by_name_when_the_app_id_differs(config):
    body = '<active-app><app id="99">netflix</app></active-app>'
    roku = client(config, FakeSession(body))
    assert roku.is_netflix_active() is True


def test_is_netflix_active_returns_none_when_unreachable(config):
    """None means 'could not ask' — the controller must not treat it as False."""
    roku = client(config, FakeSession(fail=True))
    assert roku.is_netflix_active() is None


def test_device_info_is_flattened(config):
    roku = client(config, FakeSession(DEVICE_INFO))
    info = roku.device_info()
    assert info["model-name"] == "Hisense Roku TV"
    assert info["friendly-device-name"] == "Living Room TV"


def test_device_info_returns_none_on_error(config):
    assert client(config, FakeSession(fail=True)).device_info() is None


def test_malformed_xml_does_not_raise(config):
    roku = client(config, FakeSession("<active-app><app>truncated"))
    assert roku.active_app() is None
    assert roku.is_netflix_active() is None


def test_http_error_status_is_treated_as_failure(config):
    class ErrorSession(FakeSession):
        def post(self, url, data=None, timeout=None):
            self.posts.append(url)
            return FakeResponse(status=503)

    roku = client(config, ErrorSession())
    assert roku.mute() is False
    assert roku.is_muted is False


def test_network_errors_never_escape(config):
    """Nothing in this module may raise into the main loop."""
    roku = client(config, FakeSession(fail=True))
    assert roku.keypress("VolumeMute") is False
    assert roku.mute() is False
    assert roku.unmute() is True  # already believed unmuted: no call needed
    assert roku.active_app() is None
    assert roku.device_info() is None


def test_close_closes_the_session(config):
    session = FakeSession()
    roku = client(config, session)
    roku.close()
    assert session.closed is True


# --------------------------------------------------------------------------- #
# URLs
# --------------------------------------------------------------------------- #


def test_urls_match_the_ecp_spec(config):
    session = FakeSession(DEVICE_INFO)
    roku = client(config, session)
    roku.keypress("VolumeMute")
    roku.device_info()
    roku.active_app()
    assert session.posts == ["http://192.168.0.12:8060/keypress/VolumeMute"]
    assert session.gets == [
        "http://192.168.0.12:8060/query/device-info",
        "http://192.168.0.12:8060/query/active-app",
    ]
