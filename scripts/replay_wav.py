#!/usr/bin/env python3
"""Feed a WAV file through features + detector + controller offline.

This is the tuning tool. It runs the *same* code path as the live service — the
real feature extractor, the real detector, the real state machine — with the TV
replaced by a stub, and prints one row per window plus a summary of every mute
span it would have produced.

    python scripts/replay_wav.py samples/ad_2026-08-06T20-31-02.wav
    python scripts/replay_wav.py samples/*.wav --csv /tmp/rows.csv
    python scripts/replay_wav.py sample.wav --set detection.ad_crest_delta_db=3.0

Edit config.yaml (or use --set for a one-off) and re-run until the mute spans
line up with the ads you can hear in the file.
"""

from __future__ import annotations

import argparse
import dataclasses
import logging
import sys
from pathlib import Path
from typing import Any, Iterable

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from admuter.capture import AudioWindow, wav_windows  # noqa: E402
from admuter.config import Config, ConfigError  # noqa: E402
from admuter.controller import Controller  # noqa: E402
from admuter.detector import Decision, Event, HeuristicDetector  # noqa: E402
from admuter.features import Features  # noqa: E402
from admuter.logging_setup import FeatureLogger, setup_logging  # noqa: E402

DEFAULT_CONFIG = Path(__file__).resolve().parent.parent / "config.yaml"


class StubRoku:
    """Stands in for RokuClient: records what would have been sent."""

    def __init__(self) -> None:
        self.is_muted = False
        self.actions: list[tuple[float, str]] = []
        self.timestamp = 0.0

    def mute(self) -> bool:
        if not self.is_muted:
            self.is_muted = True
            self.actions.append((self.timestamp, "MUTE"))
        return True

    def unmute(self) -> bool:
        if self.is_muted:
            self.is_muted = False
            self.actions.append((self.timestamp, "UNMUTE"))
        return True

    def is_netflix_active(self) -> bool:
        return True

    def device_info(self) -> dict[str, str]:
        return {}

    def close(self) -> None:
        pass


class TapDetector:
    """Wraps the real detector so the script can see what it was fed."""

    def __init__(self, inner: HeuristicDetector) -> None:
        self.inner = inner
        self.last_features: Features | None = None

    def update(self, features: Features, timestamp: float) -> Decision:
        self.last_features = features
        return self.inner.update(features, timestamp)

    def reset(self) -> None:
        self.inner.reset()

    def reject(self) -> None:
        self.inner.reject()


class ListSource:
    """WindowSource over an iterable of AudioWindows."""

    def __init__(self, windows: Iterable[AudioWindow]) -> None:
        self._windows = windows

    def windows(self) -> Iterable[AudioWindow]:
        return self._windows

    def stop(self) -> None:
        pass


def clock(seconds: float) -> str:
    return f"{int(seconds // 60):02d}:{seconds % 60:04.1f}"


def apply_overrides(config: Config, overrides: list[str]) -> Config:
    """Apply ``section.key=value`` overrides to a loaded config."""
    sections: dict[str, Any] = {
        name: dataclasses.asdict(getattr(config, name))
        for name in ("audio", "detection", "controller", "roku", "logging")
    }
    for override in overrides:
        if "=" not in override or "." not in override.split("=", 1)[0]:
            raise SystemExit(f"--set expects section.key=value, got {override!r}")
        dotted, raw = override.split("=", 1)
        section, _, key = dotted.partition(".")
        if section not in sections or key not in sections[section]:
            raise SystemExit(f"--set: unknown option {dotted!r}")
        current = sections[section][key]
        if isinstance(current, bool):
            value: Any = raw.strip().lower() in ("1", "true", "yes", "on")
        elif isinstance(current, int) and not isinstance(current, bool):
            value = int(raw)
        elif isinstance(current, float):
            value = float(raw)
        elif isinstance(current, list):
            value = [v for v in raw.split(",") if v]
        else:
            value = raw
        sections[section][key] = value
    return Config.from_dict(sections)


def replay(path: Path, config: Config, args: argparse.Namespace) -> int:
    detector = TapDetector(
        HeuristicDetector(config.detection, config.audio.window_seconds)
    )
    roku = StubRoku()
    feature_logger: FeatureLogger | None = None
    if args.out:
        fmt = "csv" if args.out.suffix.lower() == ".csv" else "jsonl"
        feature_logger = FeatureLogger(args.out, fmt)

    source = ListSource(wav_windows(path, config.audio.window_seconds))
    controller = Controller(source, detector, roku, config, feature_logger)

    header = (
        f"{'time':>8} {'rms':>7} {'peak':>7} {'crest':>6} {'centroid':>9} "
        f"{'sil%':>5} {'gap':>5} {'event':>10} {'conf':>5} {'state':>13} {'mute':>5}  reason"
    )
    print(f"\n=== {path} ===")
    print(header)
    print("-" * len(header))

    windows = 0
    for window in source.windows():
        roku.timestamp = window.timestamp
        decision = controller.process_window(window)
        features = detector.last_features
        windows += 1
        if features is None or decision is None:
            continue
        if args.only_changes and decision.event is Event.NO_CHANGE:
            continue
        print(
            f"{clock(window.timestamp):>8} "
            f"{features.rms_dbfs:>7.1f} "
            f"{features.peak_dbfs:>7.1f} "
            f"{features.crest_db:>6.1f} "
            f"{features.spectral_centroid_hz:>9.0f} "
            f"{features.silence_ratio * 100:>5.0f} "
            f"{decision.metrics.get('gap_seconds', 0.0):>5.2f} "
            f"{decision.event.value:>10} "
            f"{decision.confidence:>5.2f} "
            f"{controller.state.value:>13} "
            f"{'MUTED' if roku.is_muted else '-':>5}  "
            f"{decision.reason}"
        )
    controller.shutdown()

    duration = windows * config.audio.window_seconds
    print(f"\n{windows} windows ({clock(duration)} of audio)")
    if not roku.actions:
        print("no mute actions — nothing would have been muted")
        return 0

    print("mute spans that would have been applied:")
    muted_at: float | None = None
    total = 0.0
    for timestamp, action in roku.actions:
        if action == "MUTE":
            muted_at = timestamp
        elif muted_at is not None:
            total += timestamp - muted_at
            print(
                f"  {clock(muted_at)} -> {clock(timestamp)} "
                f"({timestamp - muted_at:.0f}s)"
            )
            muted_at = None
    if muted_at is not None:
        end = duration
        total += max(0.0, end - muted_at)
        print(f"  {clock(muted_at)} -> {clock(end)} (still muted at EOF)")
    share = 100.0 * total / duration if duration else 0.0
    print(f"  total muted: {total:.0f}s of {duration:.0f}s ({share:.0f}%)")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("wav", nargs="+", type=Path, help="WAV file(s) to replay")
    parser.add_argument("-c", "--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument(
        "-o", "--out", type=Path, help="write feature rows to this .jsonl/.csv file"
    )
    parser.add_argument(
        "--set",
        dest="overrides",
        action="append",
        default=[],
        metavar="section.key=value",
        help="override a config value for this run (repeatable)",
    )
    parser.add_argument(
        "--only-changes",
        action="store_true",
        help="print only windows where the detector emitted an event",
    )
    parser.add_argument(
        "--log-level",
        default="WARNING",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="controller/detector log level (default WARNING: the table is the output)",
    )
    args = parser.parse_args(argv)

    setup_logging(args.log_level)
    logging.getLogger("admuter").setLevel(args.log_level)

    try:
        config = apply_overrides(Config.load(args.config), args.overrides)
    except ConfigError as exc:
        print(f"config error: {exc}", file=sys.stderr)
        return 2

    status = 0
    for path in args.wav:
        if not path.exists():
            print(f"no such file: {path}", file=sys.stderr)
            status = 1
            continue
        status |= replay(path, config, args)
    return status


if __name__ == "__main__":
    raise SystemExit(main())
