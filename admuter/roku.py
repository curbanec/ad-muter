"""Roku External Control Protocol client.

ECP notes that shaped this module:

* ``/keypress/VolumeMute`` is a **toggle**. There is no discrete mute/unmute and
  no query that reports mute state, so we track what we believe and refuse to
  send a redundant toggle. If the believed state ever desyncs (someone used the
  remote), restarting the service while the TV is unmuted re-syncs it.
* ``/query/media-player`` is useless for ad detection — Netflix stitches ads
  server-side and the Roku reports identical state during ads and content. We
  only use ``/query/active-app`` to know whether Netflix is on screen at all.
* Nothing here raises into the main loop. Network errors are logged and
  reported as a falsy/None result; the TV being off is a normal condition.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any
from xml.etree import ElementTree

import requests

from .config import RokuConfig

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class ActiveApp:
    """Whatever the Roku says is on screen right now."""

    app_id: str
    name: str

    def matches(self, ids: list[str], names: list[str]) -> bool:
        if self.app_id and self.app_id in ids:
            return True
        lowered = self.name.strip().lower()
        return any(lowered == n.strip().lower() for n in names)


class RokuClient:
    """Thin ECP wrapper with believed-mute-state tracking."""

    def __init__(
        self,
        config: RokuConfig,
        session: requests.Session | None = None,
        dry_run: bool = False,
    ) -> None:
        self.config = config
        self.session = session if session is not None else requests.Session()
        self.dry_run = dry_run
        self._muted = config.assume_muted_at_start
        self._reachable: bool | None = None

    # ------------------------------------------------------------------ #
    # Mute state
    # ------------------------------------------------------------------ #

    @property
    def is_muted(self) -> bool:
        """What we believe the TV's mute state to be."""
        return self._muted

    def set_believed_mute_state(self, muted: bool) -> None:
        """Force the tracked state without touching the TV (re-sync escape hatch)."""
        self._muted = muted

    def mute(self) -> bool:
        """Mute the TV. No-ops (and returns True) if already believed muted."""
        if self._muted:
            return True
        if self._toggle_mute():
            self._muted = True
            return True
        return False

    def unmute(self) -> bool:
        """Unmute the TV. No-ops (and returns True) if already believed unmuted."""
        if not self._muted:
            return True
        if self._toggle_mute():
            self._muted = False
            return True
        return False

    def _toggle_mute(self) -> bool:
        if self.dry_run:
            log.info("[dry-run] would send keypress VolumeMute")
            return True
        return self.keypress("VolumeMute")

    # ------------------------------------------------------------------ #
    # ECP primitives
    # ------------------------------------------------------------------ #

    def keypress(self, key: str) -> bool:
        """POST /keypress/<key>. Returns False on any network/HTTP failure."""
        return self._post(f"/keypress/{key}")

    def device_info(self) -> dict[str, str] | None:
        """GET /query/device-info as a flat dict, or None if unavailable."""
        root = self._get_xml("/query/device-info")
        if root is None:
            return None
        return {child.tag: (child.text or "") for child in root}

    def active_app(self) -> ActiveApp | None:
        """GET /query/active-app. None means 'we could not ask', not 'nothing'."""
        root = self._get_xml("/query/active-app")
        if root is None:
            return None
        app = root.find("app")
        if app is None:
            # Screensaver-only responses omit <app>; treat as an unknown app.
            return ActiveApp(app_id="", name="")
        return ActiveApp(app_id=app.get("id", "") or "", name=(app.text or "").strip())

    def is_netflix_active(self) -> bool | None:
        """True/False if we could ask the TV, None if the query failed."""
        app = self.active_app()
        if app is None:
            return None
        return app.matches(self.config.netflix_app_ids, self.config.netflix_app_names)

    # ------------------------------------------------------------------ #
    # HTTP plumbing
    # ------------------------------------------------------------------ #

    def _url(self, path: str) -> str:
        return f"{self.config.base_url}{path}"

    def _post(self, path: str) -> bool:
        try:
            response = self.session.post(
                self._url(path), data=b"", timeout=self.config.timeout_seconds
            )
            response.raise_for_status()
        except requests.RequestException as exc:
            self._note_unreachable(path, exc)
            return False
        self._note_reachable()
        return True

    def _get_xml(self, path: str) -> ElementTree.Element | None:
        try:
            response = self.session.get(
                self._url(path), timeout=self.config.timeout_seconds
            )
            response.raise_for_status()
            body: Any = response.text
        except requests.RequestException as exc:
            self._note_unreachable(path, exc)
            return None
        self._note_reachable()
        try:
            return ElementTree.fromstring(body)
        except ElementTree.ParseError as exc:
            log.warning("roku %s returned unparseable XML: %s", path, exc)
            return None

    def _note_unreachable(self, path: str, exc: Exception) -> None:
        # Log the first failure loudly, then stay quiet until it recovers — a TV
        # that is simply switched off should not fill the journal.
        if self._reachable is not False:
            log.warning("roku %s unreachable (%s): %s", self.config.host, path, exc)
        else:
            log.debug("roku %s still unreachable (%s): %s", self.config.host, path, exc)
        self._reachable = False

    def _note_reachable(self) -> None:
        if self._reachable is False:
            log.info("roku %s reachable again", self.config.host)
        self._reachable = True

    def close(self) -> None:
        try:
            self.session.close()
        except Exception:  # pragma: no cover - defensive
            pass
