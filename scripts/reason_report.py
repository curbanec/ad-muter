#!/usr/bin/env python3
"""Explain what the detector was thinking, from a live feature log.

    python scripts/reason_report.py logs/features.jsonl
    python scripts/reason_report.py logs/features.jsonl --from 20:41 --to 20:45

One line per window is unreadable at 3600 rows an hour. This collapses runs of
identical (state, reason) into single lines, so an evening becomes a page and a
missed break becomes one line saying which half of the test failed.

Reasons that differ only in their numbers -- "ad continues (12s)" and "ad
continues (13s)" -- are one reason, not two.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from admuter.config import Config, ConfigError  # noqa: E402

DEFAULT_CONFIG = Path(__file__).resolve().parent.parent / "config.yaml"

NUMBERS = re.compile(r"[-+]?\d+(?:\.\d+)?")

LEGEND = {
    "not armed":
        "the app gate had detection off",
    "baseline warming up":
        "the detector was reset recently (app switch or capture restart)",
    "transition cue armed, waiting for ad profile":
        "a seam was seen, but the loudness/crest check said no",
    "ad profile without a transition cue — holding":
        "the audio looked like an ad, but there was no recent seam",
    "content":
        "neither the seam nor the loudness check fired",
    "ad continues (Ns)":
        "muted, and the audio still looks like an ad",
    "ad profile lapsed (N/N windows, Ns elapsed)":
        "muted, counting down to unmute because the audio stopped looking like an ad",
}


def generalise(reason: str) -> str:
    """"ad continues (12s)" and "(13s)" are the same reason."""
    return NUMBERS.sub("N", reason or "")


def parse_clock(value: str, first_wall: datetime | None) -> float | None:
    """Accept HH:MM wall time or plain seconds; return seconds from the start."""
    if value is None:
        return None
    if ":" in value:
        if first_wall is None:
            raise SystemExit("--from/--to given as HH:MM but the log has no wall_time")
        hour, minute = (int(p) for p in value.split(":")[:2])
        target = first_wall.replace(hour=hour, minute=minute, second=0, microsecond=0)
        return (target - first_wall).total_seconds()
    return float(value)


def load(path: Path) -> list[dict]:
    rows = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                continue  # a half-written final line while the service runs
    return rows


def which_check_failed(row: dict, crest_bar: float) -> str:
    """For 'cue armed, waiting', say whether loudness, crest or both said no."""
    smoothed = row.get("smoothed_rms_dbfs")
    bar = row.get("loudness_threshold_db")
    crest = row.get("crest_delta_db")
    failures = []
    if smoothed is not None and bar is not None and smoothed < bar:
        failures.append(f"loudness {smoothed:.1f} < {bar:.1f} dBFS")
    if crest is not None and crest < crest_bar:
        failures.append(f"crest {crest:.1f} < {crest_bar:.1f} dB")
    if not failures:
        return ""
    return " and ".join(failures)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("log", type=Path, help="logs/features.jsonl")
    parser.add_argument("-c", "--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--from", dest="start", metavar="HH:MM|SECONDS")
    parser.add_argument("--to", dest="end", metavar="HH:MM|SECONDS")
    parser.add_argument("--min-seconds", type=float, default=0.0,
                        help="hide runs shorter than this")
    args = parser.parse_args(argv)

    if not args.log.exists():
        print(f"error: no such log: {args.log}", file=sys.stderr)
        return 1
    rows = load(args.log)
    if not rows:
        print("log is empty")
        return 0

    try:
        crest_bar = Config.load(args.config).detection.ad_crest_delta_db
    except (ConfigError, OSError):
        crest_bar = 0.0

    def wall_of(row):
        raw = row.get("wall_time")
        if not raw:
            return None
        try:
            return datetime.fromisoformat(raw)
        except ValueError:
            return None

    first_wall = wall_of(rows[0])
    base = rows[0].get("timestamp", 0.0)
    lo = parse_clock(args.start, first_wall)
    hi = parse_clock(args.end, first_wall)

    kept = []
    for row in rows:
        offset = row.get("timestamp", 0.0) - base
        if lo is not None and offset < lo:
            continue
        if hi is not None and offset > hi:
            continue
        kept.append((offset, row))
    if not kept:
        print("nothing in that range")
        return 0

    # --- collapse into runs ------------------------------------------------ #
    runs = []
    for offset, row in kept:
        key = (row.get("state", ""), generalise(row.get("reason", "")),
               bool(row.get("muted")))
        if runs and runs[-1]["key"] == key:
            runs[-1]["end"] = offset
            runs[-1]["rows"].append(row)
        else:
            runs.append({"key": key, "start": offset, "end": offset,
                         "wall": wall_of(row), "rows": [row]})

    window = 1.0
    if len(kept) > 1:
        window = max(0.001, kept[1][0] - kept[0][0])

    print(f"{args.log}  —  {len(kept)} windows, "
          f"{(kept[-1][0] - kept[0][0] + window) / 60:.1f} min\n")
    header = f"{'start':>10}{'dur':>8}  {'state':<14}{'mute':<6}reason"
    print(header)
    print("-" * max(len(header), 74))

    totals: dict[str, float] = defaultdict(float)
    muted_seconds = 0.0
    for run in runs:
        duration = run["end"] - run["start"] + window
        state, reason, muted = run["key"]
        totals[reason] += duration
        if muted:
            muted_seconds += duration
        if duration < args.min_seconds:
            continue
        start = (run["wall"].strftime("%H:%M:%S") if run["wall"]
                 else f"{run['start']:.0f}s")
        detail = ""
        if reason.startswith("transition cue armed"):
            reasons = {which_check_failed(r, crest_bar) for r in run["rows"]}
            reasons.discard("")
            if reasons:
                detail = "  [" + "; ".join(sorted(reasons)[:2]) + "]"
        print(f"{start:>10}{duration:>7.0f}s  {state:<14}"
              f"{'MUTED' if muted else '':<6}{reason}{detail}")

    # --- totals ------------------------------------------------------------ #
    span = kept[-1][0] - kept[0][0] + window
    print(f"\n{'seconds':>10}  {'%':>5}  reason")
    print("-" * 74)
    for reason, seconds in sorted(totals.items(), key=lambda kv: -kv[1]):
        print(f"{seconds:>10.0f}  {100 * seconds / span:>4.0f}%  {reason}")
    print("-" * 74)
    print(f"{muted_seconds:>10.0f}  {100 * muted_seconds / span:>4.0f}%  MUTED")

    seen = {r for r in totals if r in LEGEND}
    if seen:
        print("\nwhat these mean:")
        for reason in sorted(seen):
            print(f"  {reason}\n      {LEGEND[reason]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
