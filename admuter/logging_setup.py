"""Console logging plus the optional per-window feature dump.

Everything goes to stdout: under systemd, journald picks it up.

The feature log is the point of Phase 1 that pays off in Phase 2 — every window
lands in a JSONL/CSV row together with the decision the heuristics made, which
is most of a labelled training set once you annotate the ad spans.
"""

from __future__ import annotations

import csv
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


class FeatureLogger:
    """Appends one row per window to JSONL or CSV."""

    def __init__(self, path: str | Path, fmt: str = "jsonl") -> None:
        fmt = fmt.lower()
        if fmt not in ("jsonl", "csv"):
            raise ValueError(f"unsupported feature log format: {fmt!r}")
        self.path = Path(path)
        self.format = fmt
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._fh: TextIO = self.path.open("a", encoding="utf-8", newline="")
        self._writer: csv.DictWriter | None = None
        self._closed = False

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
