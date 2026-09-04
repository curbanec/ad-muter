#!/usr/bin/env python3
"""Score the real detector + controller against annotated chunks.

Window-level accuracy is the wrong lens for this system. The controller only
mutes after ``confirm_windows`` consecutive ad-like windows and holds the mute
through ``min_ad_seconds``, so isolated bad windows never reach the TV. What
matters is the mute spans that come out the other end.

This runs the actual HeuristicDetector and Controller over a session's chunks —
in timestamp order, with state carried across chunk boundaries the way the live
service experiences it — and compares the resulting mute spans to the labels.

    python scripts/score_detector.py -i samples/movie
    python scripts/score_detector.py -i samples/movie --set detection.baseline_alpha=0.01
    python scripts/score_detector.py -i samples/movie --sweep detection.ad_crest_delta_db=-99,0,2

Three numbers decide whether it works:

* **breaks caught** — how many labelled breaks produced a mute at all.
* **latency** — seconds from break start to mute. Your budget is 1-2s.
* **false mute seconds** — audio muted that was not an ad. This is the one that
  ruins the experience; a missed ad is merely annoying.
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass, field
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from admuter.capture import AudioWindow, wav_windows  # noqa: E402
from admuter.config import Config, ConfigError  # noqa: E402
from admuter.controller import Controller  # noqa: E402
from admuter.detector import HeuristicDetector  # noqa: E402
from admuter.logging_setup import setup_logging  # noqa: E402

from build_dataset import (  # noqa: E402
    AD_LABELS,
    DatasetError,
    Span,
    pair_chunks,
    parse_labels,
)

DEFAULT_CONFIG = Path(__file__).resolve().parent.parent / "config.yaml"
# A mute that starts slightly before a break, or lingers slightly after, is not
# really a mistake — the seam itself is dead air either way.
EDGE_GRACE_SECONDS = 2.0


def clock(seconds: float) -> str:
    return f"{int(seconds // 60):02d}:{seconds % 60:04.1f}"


@dataclass
class RecordingRoku:
    """Stands in for the TV; records when a mute would have started/stopped."""

    is_muted: bool = False
    timestamp: float = 0.0
    spans: list[list[float]] = field(default_factory=list)

    def mute(self) -> bool:
        if not self.is_muted:
            self.is_muted = True
            self.spans.append([self.timestamp, self.timestamp])
        return True

    def unmute(self) -> bool:
        if self.is_muted:
            self.is_muted = False
            self.spans[-1][1] = self.timestamp
        return True

    def is_netflix_active(self) -> bool:
        return True

    def close(self) -> None:
        pass


class SessionSource:
    """Every chunk's windows, renumbered onto one continuous session clock."""

    def __init__(self, pairs, window_seconds: float, roku: RecordingRoku) -> None:
        self.pairs = pairs
        self.window_seconds = window_seconds
        self.roku = roku
        self.duration = 0.0

    def windows(self):
        elapsed = 0.0
        index = 0
        for wav, _ in self.pairs:
            consumed = 0.0
            for window in wav_windows(wav, self.window_seconds):
                t = elapsed + window.timestamp
                self.roku.timestamp = t
                yield AudioWindow(
                    samples=window.samples,
                    sample_rate=window.sample_rate,
                    index=index,
                    timestamp=t,
                    # Only the very first window is a genuine stream start; a
                    # chunk boundary is a recording artifact, not a restart.
                    stream_restarted=index == 0,
                )
                index += 1
                consumed = window.timestamp + self.window_seconds
            elapsed += consumed
        self.duration = elapsed

    def stop(self) -> None:
        pass


def session_spans(pairs, window_seconds: float) -> list[Span]:
    """Label spans shifted onto the same continuous session clock."""
    spans: list[Span] = []
    elapsed = 0.0
    for wav, label_path in pairs:
        for span in parse_labels(label_path):
            if span.label in AD_LABELS:
                spans.append(Span(span.start + elapsed, span.end + elapsed, "ad"))
        with __import__("wave").open(str(wav), "rb") as w:
            elapsed += w.getnframes() / w.getframerate()
    # Stitch spans that meet at a chunk boundary back into one break.
    merged: list[Span] = []
    for span in sorted(spans, key=lambda s: s.start):
        if merged and span.start - merged[-1].end < 2.0:
            merged[-1] = Span(merged[-1].start, span.end, "ad")
        else:
            merged.append(span)
    return merged


def overlap(a: Span, b: tuple[float, float]) -> float:
    return max(0.0, min(a.end, b[1]) - max(a.start, b[0]))


def score(pairs, config: Config) -> dict:
    roku = RecordingRoku()
    source = SessionSource(pairs, config.audio.window_seconds, roku)
    detector = HeuristicDetector(config.detection, config.audio.window_seconds)
    controller = Controller(source, detector, roku, config)
    controller.run()

    duration = source.duration
    mutes = [(s[0], s[1] if s[1] > s[0] else duration) for s in roku.spans]
    breaks = session_spans(pairs, config.audio.window_seconds)

    ad_seconds = sum(b.end - b.start for b in breaks)
    muted_seconds = sum(e - s for s, e in mutes)
    covered = sum(sum(overlap(b, m) for m in mutes) for b in breaks)

    false_muted = 0.0
    for s, e in mutes:
        padded = [Span(b.start - EDGE_GRACE_SECONDS, b.end + EDGE_GRACE_SECONDS, "ad")
                  for b in breaks]
        inside = sum(overlap(p, (s, e)) for p in padded)
        false_muted += max(0.0, (e - s) - inside)

    caught = []
    for b in breaks:
        hits = [m for m in mutes if overlap(b, m) > 0]
        caught.append((b, min((m[0] for m in hits), default=None)))

    return {
        "duration": duration,
        "breaks": caught,
        "mutes": mutes,
        "ad_seconds": ad_seconds,
        "muted_seconds": muted_seconds,
        "covered": covered,
        "false_muted": false_muted,
    }


def report(result: dict, label: str = "") -> None:
    if label:
        print(f"\n=== {label} ===")
    breaks = result["breaks"]
    hit = sum(1 for _, start in breaks if start is not None)
    print(f"  breaks caught: {hit}/{len(breaks)}")
    for span, start in breaks:
        if start is None:
            print(f"    {clock(span.start)}-{clock(span.end)}  MISSED")
        else:
            print(
                f"    {clock(span.start)}-{clock(span.end)}  muted at "
                f"{clock(start)}  (latency {start - span.start:+.0f}s)"
            )
    ad = result["ad_seconds"]
    cov = 100.0 * result["covered"] / ad if ad else 0.0
    print(f"  ad audio muted:    {result['covered']:.0f}s of {ad:.0f}s ({cov:.0f}%)")
    print(f"  FALSE MUTE:        {result['false_muted']:.0f}s of content")
    extra = [m for m in result["mutes"]
             if all(not (m[0] < s.end and m[1] > s.start) for s, _ in breaks)]
    for s, e in extra:
        print(f"    spurious mute {clock(s)}-{clock(e)} ({e - s:.0f}s)")


def apply_overrides(config: Config, overrides: list[str]) -> Config:
    import dataclasses
    sections = {n: dataclasses.asdict(getattr(config, n))
                for n in ("audio", "detection", "controller", "roku", "logging")}
    for override in overrides:
        dotted, _, raw = override.partition("=")
        section, _, key = dotted.partition(".")
        if section not in sections or key not in sections[section]:
            raise SystemExit(f"--set: unknown option {dotted!r}")
        current = sections[section][key]
        if isinstance(current, bool):
            value = raw.strip().lower() in ("1", "true", "yes", "on")
        elif isinstance(current, int) and not isinstance(current, bool):
            value = int(raw)
        elif isinstance(current, float):
            value = float(raw)
        else:
            value = raw
        sections[section][key] = value
    return Config.from_dict(sections)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("-i", "--indir", type=Path, default=Path("."))
    parser.add_argument("-c", "--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--set", dest="overrides", action="append", default=[],
                        metavar="section.key=value")
    parser.add_argument("--sweep", metavar="section.key=v1,v2,v3",
                        help="score once per value of one setting")
    parser.add_argument("--log-level", default="ERROR",
                        choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    args = parser.parse_args(argv)
    setup_logging(args.log_level)

    try:
        base = apply_overrides(Config.load(args.config), args.overrides)
        pairs = pair_chunks(args.indir)
    except (ConfigError, DatasetError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    print(f"{len(pairs)} chunks from {args.indir}")
    if not args.sweep:
        report(score(pairs, base), "current config")
        return 0

    dotted, _, values = args.sweep.partition("=")
    for value in values.split(","):
        config = apply_overrides(base, [f"{dotted}={value}"])
        report(score(pairs, config), f"{dotted}={value}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
