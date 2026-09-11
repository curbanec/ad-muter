#!/usr/bin/env python3
"""Measure the seams: are they a door the detector could actually use?

    python scripts/seam_report.py --loso ~/Movies/ad-muter-annotated
    python scripts/seam_report.py --loso ~/Movies/ad-muter-annotated --before 10 --after 3

The proposed redesign treats a seam as a door and loudness as the sign telling
you which side you are now on: enter a break at a seam where the audio steps up,
hold through the loud-to-loud seams between spots, leave at the seam where it
drops back to show level. That only works if three things are true, and this
measures all three without touching the detector:

* breaks actually begin and end at a seam (coverage),
* the loudness step at a break-start seam separates from the step at an
  ordinary content seam (otherwise every door looks alike),
* audio after an inside-break seam stays loud relative to show level, so
  "still in the break" is distinguishable from "back to the programme".

Read-only. Nothing here changes detection; it reuses the live feature path and
the same SeamTracker the detector uses, so a seam counted here is a seam the
detector could have acted on.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import statistics
import subprocess
import sys
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from admuter.capture import wav_windows  # noqa: E402
from admuter.config import Config, ConfigError  # noqa: E402
from admuter.detector import SeamTracker  # noqa: E402
from admuter.features import compute_features  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))

from build_dataset import DatasetError, pair_chunks, read_session  # noqa: E402
from score_detector import discover_sessions, session_spans, wav_duration  # noqa: E402

DEFAULT_CONFIG = Path(__file__).resolve().parent.parent / "config.yaml"
TOLERANCE = 2.0          # a seam this close to a labelled edge IS that edge
STEP_THRESHOLDS = (1.0, 2.0, 3.0, 4.0, 5.0)

BREAK_START, BREAK_END, INSIDE, CONTENT = (
    "break start", "break end", "inside break", "content",
)


@dataclass
class Seam:
    session_id: str
    at: float
    gap: float
    category: str
    before: float | None
    after: float | None
    step: float | None
    after_vs_show: float | None   # after - pre_break_level, for end/inside seams


def median(values):
    return statistics.median(values) if values else None


def percentile(values, q: float):
    """q in 0..100. Nearest-rank; the samples here are far too few for interpolation."""
    if not values:
        return None
    ordered = sorted(values)
    k = max(0, min(len(ordered) - 1, int(round(q / 100.0 * (len(ordered) - 1)))))
    return ordered[k]


def scan_session(indir: Path, config: Config, before_n: int, after_n: int):
    """Replay one session and return (session_id, seams, breaks, duration)."""
    pairs = pair_chunks(indir)
    session_id = read_session(indir).get("session_id") or indir.resolve().name
    detection = config.detection
    window_seconds = config.audio.window_seconds

    tracker = SeamTracker(detection.min_gap_seconds, detection.max_gap_seconds)
    stamps: list[float] = []
    levels: list[float] = []       # rms_dbfs, non-silent windows only
    voiced_at: list[float] = []
    seam_times: list[tuple[float, float]] = []

    elapsed = 0.0
    for wav, _ in pairs:
        for window in wav_windows(wav, window_seconds):
            t = elapsed + window.timestamp
            features = compute_features(
                window.samples, window.sample_rate,
                silence_dbfs=detection.silence_dbfs,
                frame_seconds=config.audio.frame_seconds,
            )
            stamps.append(t)
            if not features.is_silence:
                voiced_at.append(t)
                levels.append(features.rms_dbfs)
            gap = tracker.gap(features)
            if tracker.qualifies(gap):
                seam_times.append((t, gap))
        # Exact clock, matching session_spans (see Prompt 1b).
        elapsed += wav_duration(wav)

    breaks = session_spans(pairs, window_seconds)

    def level_before(t: float, n: int):
        picked = [lvl for at, lvl in zip(voiced_at, levels) if at < t][-n:]
        return median(picked)

    def level_after(t: float, n: int):
        picked = [lvl for at, lvl in zip(voiced_at, levels) if at > t][:n]
        return median(picked)

    # Pre-break show level: the "before" of the break's own start seam when it
    # has one, else the windows preceding the labelled start.
    pre_break: dict[int, float | None] = {}
    for i, span in enumerate(breaks):
        start_seam = next(
            (t for t, _ in seam_times if abs(t - span.start) <= TOLERANCE), None
        )
        anchor = start_seam if start_seam is not None else span.start
        pre_break[i] = level_before(anchor, before_n)

    seams: list[Seam] = []
    for t, gap in seam_times:
        category, index = CONTENT, None
        for i, span in enumerate(breaks):
            if abs(t - span.start) <= TOLERANCE:
                category, index = BREAK_START, i
                break
            if abs(t - span.end) <= TOLERANCE:
                category, index = BREAK_END, i
                break
            if span.start < t < span.end:
                category, index = INSIDE, i
                break
        before = level_before(t, before_n)
        after = level_after(t, after_n)
        step = (after - before) if (before is not None and after is not None) else None
        after_vs_show = None
        if category in (BREAK_END, INSIDE) and index is not None:
            show = pre_break.get(index)
            if show is not None and after is not None:
                after_vs_show = after - show
        seams.append(Seam(session_id, t, gap, category, before, after, step,
                          after_vs_show))

    return session_id, seams, breaks, (stamps[-1] + window_seconds if stamps else 0.0)


def coverage(breaks, seams, edge: str):
    """How many labelled break edges have a seam within tolerance."""
    hits = 0
    for span in breaks:
        target = span.start if edge == "start" else span.end
        if any(abs(s.at - target) <= TOLERANCE for s in seams):
            hits += 1
    return hits, len(breaks)


def distribution(seams, attr: str, category: str):
    values = [getattr(s, attr) for s in seams
              if s.category == category and getattr(s, attr) is not None]
    return {
        "n": len(values),
        "p10": percentile(values, 10),
        "median": median(values),
        "p90": percentile(values, 90),
    }


def fmt(value, width=7, places=1):
    return f"{value:>{width}.{places}f}" if value is not None else f"{'-':>{width}}"


def report_session(session_id, seams, breaks, duration, ad_seconds):
    starts = coverage(breaks, seams, "start")
    ends = coverage(breaks, seams, "end")
    content_hours = max(duration - ad_seconds, 0.0) / 3600.0
    content_seams = [s for s in seams if s.category == CONTENT]

    print(f"\n  [{session_id}]  {duration / 60:.0f} min, {len(breaks)} breaks, "
          f"{len(seams)} seams")
    print(f"    break starts with a seam: {starts[0]}/{starts[1]}"
          f"    break ends with a seam: {ends[0]}/{ends[1]}")

    print(f"    {'category':<14}{'n':>4}{'step p10':>10}{'step med':>10}{'step p90':>10}"
          f"{'vs show med':>13}")
    for category in (BREAK_START, INSIDE, BREAK_END, CONTENT):
        d = distribution(seams, "step", category)
        v = distribution(seams, "after_vs_show", category)
        print(f"    {category:<14}{d['n']:>4}{fmt(d['p10'], 10)}{fmt(d['median'], 10)}"
              f"{fmt(d['p90'], 10)}{fmt(v['median'], 13)}")

    if content_hours > 0:
        rates = [f"≥{t:.0f}dB {sum(1 for s in content_seams if s.step is not None and s.step >= t) / content_hours:.0f}"
                 for t in STEP_THRESHOLDS]
        print(f"    content seams/content-hr: {len(content_seams) / content_hours:.0f}"
              f"   of which  " + "  ".join(rates))

    start_seams = [s for s in seams if s.category == BREAK_START]
    if start_seams:
        hits = [f"≥{t:.0f}dB {sum(1 for s in start_seams if s.step is not None and s.step >= t)}/{len(breaks)}"
                for t in STEP_THRESHOLDS]
        print(f"    break starts by step:     " + "  ".join(hits))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("-i", "--indir", type=Path, action="append", dest="indirs")
    parser.add_argument("--loso", type=Path, metavar="PARENT")
    parser.add_argument("-c", "--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--before", type=int, default=10,
                        help="non-silent windows before a seam for the 'before' level")
    parser.add_argument("--after", type=int, default=3,
                        help="non-silent windows after a seam for the 'after' level")
    parser.add_argument("--out", type=Path, default=Path("results/seams"),
                        help="directory for the stamped JSON")
    args = parser.parse_args(argv)

    try:
        config = Config.load(args.config)
    except ConfigError as exc:
        print(f"config error: {exc}", file=sys.stderr)
        return 2

    dirs = list(args.indirs or [])
    if args.loso:
        try:
            dirs += discover_sessions(args.loso)
        except DatasetError as exc:
            print(f"error: {exc}", file=sys.stderr)
            return 1
    if not dirs:
        print("error: give -i or --loso", file=sys.stderr)
        return 2

    d = config.detection
    print(f"seam = {d.min_gap_seconds}-{d.max_gap_seconds}s below "
          f"{d.silence_dbfs} dBFS; before={args.before} after={args.after} windows; "
          f"edge tolerance ±{TOLERANCE:.0f}s")

    pooled: list[Seam] = []
    pooled_breaks = []
    total_duration = total_ad = 0.0
    per_session = []

    for indir in dirs:
        try:
            session_id, seams, breaks, duration = scan_session(
                indir, config, args.before, args.after
            )
        except DatasetError as exc:
            print(f"warning: skipping {indir.name} — {exc}", file=sys.stderr)
            continue
        ad_seconds = sum(b.end - b.start for b in breaks)
        report_session(session_id, seams, breaks, duration, ad_seconds)
        pooled += seams
        pooled_breaks += breaks
        total_duration += duration
        total_ad += ad_seconds
        per_session.append({
            "session_id": session_id,
            "minutes": round(duration / 60, 1),
            "breaks": len(breaks),
            "seams": len(seams),
            "starts_with_seam": coverage(breaks, seams, "start"),
            "ends_with_seam": coverage(breaks, seams, "end"),
            "step": {c: distribution(seams, "step", c)
                     for c in (BREAK_START, INSIDE, BREAK_END, CONTENT)},
            "after_vs_show": {c: distribution(seams, "after_vs_show", c)
                              for c in (INSIDE, BREAK_END)},
        })

    print("\n" + "=" * 78)
    print("POOLED")
    report_session("all sessions", pooled, pooled_breaks, total_duration, total_ad)

    args.out.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    try:
        sha = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True,
                             text=True, timeout=5,
                             cwd=Path(__file__).resolve().parent.parent).stdout.strip()
    except (OSError, subprocess.SubprocessError):
        sha = None
    payload = {
        "timestamp": stamp,
        "git_sha": sha or None,
        "config_path": str(args.config),
        "config_sha256": hashlib.sha256(args.config.read_bytes()).hexdigest(),
        "seam_definition": {
            "min_gap_seconds": d.min_gap_seconds,
            "max_gap_seconds": d.max_gap_seconds,
            "silence_dbfs": d.silence_dbfs,
            "before_windows": args.before,
            "after_windows": args.after,
            "edge_tolerance_seconds": TOLERANCE,
        },
        "sessions": per_session,
        "seams": [asdict(s) for s in pooled],
    }
    path = args.out / f"{stamp}.json"
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(f"\nwrote {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
