"""Console logging plus the optional per-window feature dump.

Everything goes to stdout: under systemd, journald picks it up.

The feature log is the point of Phase 1 that pays off in Phase 2 — every window
lands in a JSONL/CSV row together with the decision the heuristics made, which
is most of a labelled training set once you annotate the ad spans.
"""

from __future__ import annotations

import csv
from datetime import datetime
import json
import logging
import sys
from pathlib import Path
from typing import Any, TextIO

from .config import LoggingConfig
from .detector import Decision
from .features import Features

LOG_FORMAT = "%(asctime)s %(levelname)-7s %(name)-18s %(message)s"
DATE_FORMAT = "%Y-%m-%d %H:%M:%S"


def setup_logging(level: str = "INFO", stream: TextIO | None = None) -> None:
    """Configure root logging to stdout. Safe to call more than once."""
    handler = logging.StreamHandler(stream if stream is not None else sys.stdout)
    handler.setFormatter(logging.Formatter(LOG_FORMAT, datefmt=DATE_FORMAT))
    root = logging.getLogger()
    for existing in list(root.handlers):
        root.removeHandler(existing)
    root.addHandler(handler)
    root.setLevel(getattr(logging, level.upper(), logging.INFO))
    # requests/urllib3 chatter is noise at DEBUG once per second.
    logging.getLogger("urllib3").setLevel(logging.WARNING)


_featurelog = logging.getLogger(__name__)

WALL_TIME_FIELD = "wall_time"


class FeatureLogger:
    """Appends one row per window to JSONL or CSV."""

    def __init__(self, path: str | Path, fmt: str = "jsonl") -> None:
        fmt = fmt.lower()
        if fmt not in ("jsonl", "csv"):
            raise ValueError(f"unsupported feature log format: {fmt!r}")
        self.path = Path(path)
        self.format = fmt
        self.path.parent.mkdir(parents=True, exist_ok=True)
        if fmt == "csv":
            self._rotate_if_header_is_stale()
        self._fh: TextIO = self.path.open("a", encoding="utf-8", newline="")
        self._writer: csv.DictWriter | None = None
        self._closed = False

    def _rotate_if_header_is_stale(self) -> None:
        """Move an old CSV aside if its header predates a column we now write.

        Appending new columns to a file whose header lacks them writes values
        under the wrong headings, which is worse than losing the file: the data
        looks fine and is silently wrong. Rotating keeps the old rows readable
        and starts a clean file.
        """
        if not self.path.exists() or self.path.stat().st_size == 0:
            return
        try:
            with self.path.open("r", encoding="utf-8", newline="") as handle:
                header = next(csv.reader(handle), [])
        except OSError:
            return
        if WALL_TIME_FIELD in header:
            return
        rotated = self.path.with_name(
            f"{self.path.stem}.pre-{WALL_TIME_FIELD}{self.path.suffix}"
        )
        n = 1
        while rotated.exists():
            rotated = self.path.with_name(
                f"{self.path.stem}.pre-{WALL_TIME_FIELD}.{n}{self.path.suffix}"
            )
            n += 1
        try:
            self.path.rename(rotated)
            _featurelog.info("feature log header predates %s; moved old rows to %s",
                     WALL_TIME_FIELD, rotated.name)
        except OSError:  # pragma: no cover - defensive
            _featurelog.warning("could not rotate %s; %s will be missing from new rows",
                        self.path, WALL_TIME_FIELD)

    def log(
        self,
        *,
        index: int,
        timestamp: float,
        features: Features,
        decision: Decision | None,
        state: str,
        muted: bool,
        label: str = "",
    ) -> None:
        if self._closed:
            return
        row: dict[str, Any] = {
            "index": index,
            # timestamp is time.monotonic(): fine for arithmetic, useless for
            # lining a row up with "the break around 8:15". wall_time is what
            # a human reads the log against.
            WALL_TIME_FIELD: datetime.now().astimezone().replace(
                microsecond=0
            ).isoformat(),
            "timestamp": round(timestamp, 3),
            "state": state,
            "muted": muted,
            "label": label,
        }
        row.update(features.as_dict())
        if decision is not None:
            row["event"] = decision.event.value
            row["ad_profile"] = decision.ad_profile
            row["confidence"] = decision.confidence
            row["reason"] = decision.reason
            row.update({k: round(v, 4) for k, v in decision.metrics.items()})
        else:
            row["event"] = ""
            row["ad_profile"] = False
            row["confidence"] = 0.0
            row["reason"] = "not armed"

        if self.format == "jsonl":
            self._fh.write(json.dumps(row, default=str) + "\n")
        else:
            if self._writer is None:
                self._writer = csv.DictWriter(self._fh, fieldnames=list(row))
                if self.path.stat().st_size == 0:
                    self._writer.writeheader()
            # Rows written after a header exists must not gain/lose columns.
            self._writer.writerow(
                {k: row.get(k, "") for k in self._writer.fieldnames}
            )
        self._fh.flush()

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        try:
            self._fh.close()
        except Exception:  # pragma: no cover - defensive
            pass

    def __enter__(self) -> "FeatureLogger":
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.close()


def build_feature_logger(config: LoggingConfig) -> FeatureLogger | None:
    """Construct a FeatureLogger if enabled in config, else None."""
    if not config.feature_log_enabled:
        return None
    return FeatureLogger(config.feature_log_path, config.feature_log_format)
