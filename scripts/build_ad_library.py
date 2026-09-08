#!/usr/bin/env python3
"""Build the fingerprint library from annotated recordings.

    python scripts/build_ad_library.py --loso ~/Movies -o models/ads.json
    python scripts/build_ad_library.py --loso ~/Movies --report

Fingerprinting can only recognise a spot it has already heard, so an empty
library does nothing at all. This seeds one from the labelled ad spans you have
already annotated -- the fastest way to a library that is worth carrying.

The live service grows it further on its own: ``RepeatDetector`` notices audio
recurring at unrelated times, and since content never repeats (0.0% of the
annotated corpus at 12-word granularity, against 30.1% of ad audio) a repeat is
essentially always an ad. That path logs candidates rather than promoting them,
because a repeated three-second block marks where a spot recurs, not where it
starts and stops, and an entry with the wrong boundaries would hold mutes over
the wrong span.

``--report`` runs the honest measurement instead of writing anything: it walks
the spans in order, matching each against only what came before, exactly as the
live service would. That is the number to trust -- building a library from every
span and then querying it with those same spans measures nothing.
"""

from __future__ import annotations

import argparse
import sys
import wave
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from admuter.fingerprint import (  # noqa: E402
    HOP_SIZE,
    QUERY_FRAMES,
    SAMPLE_RATE,
    Ad,
    Fingerprinter,
    FingerprintIndex,
)

from build_dataset import AD_LABELS, DatasetError, pair_chunks, parse_labels  # noqa: E402

MIN_AD_SECONDS = 10.0


def read_mono_16k(wav: Path, start: float, end: float) -> np.ndarray:
    with wave.open(str(wav), "rb") as handle:
        rate, channels = handle.getframerate(), handle.getnchannels()
        handle.setpos(min(int(start * rate), handle.getnframes()))
        count = min(int((end - start) * rate), handle.getnframes() - handle.tell())
        raw = handle.readframes(max(0, count))
    if not raw:
        return np.zeros(0, dtype=np.float32)
    audio = np.frombuffer(raw, dtype="<i2").astype(np.float32) / 32768.0
    if channels > 1:
        audio = audio.reshape(-1, channels).mean(axis=1)
    factor = max(1, int(round(rate / SAMPLE_RATE)))
    usable = (len(audio) // factor) * factor
    if usable == 0:
        return np.zeros(0, dtype=np.float32)
    return audio[:usable].reshape(-1, factor).mean(axis=1).astype(np.float32)


def collect(dirs: list[Path]) -> list[tuple[str, Path, float, float]]:
    spans = []
    for d in dirs:
        try:
            pairs = pair_chunks(d)
        except DatasetError as exc:
            print(f"warning: skipping {d.name} — {exc}", file=sys.stderr)
            continue
        for wav, label_path in pairs:
            for span in sorted(parse_labels(label_path), key=lambda s: s.start):
                if span.label in AD_LABELS and span.end - span.start >= MIN_AD_SECONDS:
                    spans.append((d.name, wav, span.start, span.end))
    return spans


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("-i", "--indir", type=Path, action="append", dest="indirs")
    parser.add_argument("--loso", type=Path, metavar="PARENT")
    parser.add_argument("-o", "--out", type=Path, default=Path("models/ads.json"))
    parser.add_argument("--report", action="store_true",
                        help="measure recognition rate instead of writing a library")
    args = parser.parse_args(argv)

    dirs = list(args.indirs or [])
    if args.loso:
        dirs += [d for d in sorted(args.loso.iterdir())
                 if d.is_dir() and any(d.glob("*.wav"))]
    if not dirs:
        print("error: give -i or --loso", file=sys.stderr)
        return 2

    spans = collect(dirs)
    print(f"{len(spans)} labelled ad spans over {MIN_AD_SECONDS:.0f}s "
          f"from {len(dirs)} sessions")

    index = FingerprintIndex()
    recognised = matched_seconds = total_seconds = 0

    for name, wav, start, end in spans:
        audio = read_mono_16k(wav, start, end)
        if audio.size == 0:
            continue
        hashes = Fingerprinter().feed(audio)
        duration = len(hashes) * HOP_SIZE / SAMPLE_RATE
        total_seconds += duration

        # Query BEFORE adding: only what was already heard may match, which is
        # the only ordering that says anything about live performance.
        hit = None
        for offset in range(0, max(1, len(hashes) - QUERY_FRAMES), QUERY_FRAMES // 2):
            hit = index.query(hashes[offset:offset + QUERY_FRAMES])
            if hit is not None:
                break
        if hit is not None:
            recognised += 1
            matched_seconds += duration
            if args.report:
                print(f"  recognised {name[:24]:<24}{start:8.1f}s  as {hit.ad_id[:30]:<30}"
                      f"  BER {hit.bit_error_rate:.3f}")

        index.add(Ad(ad_id=f"{name}@{start:.0f}", hashes=hashes, label=name))

    print(f"\n{recognised}/{len(spans)} spans recognised from an earlier airing")
    if total_seconds:
        print(f"{matched_seconds:.0f}s of {total_seconds:.0f}s of ad audio "
              f"({matched_seconds / total_seconds:.1%})")

    if args.report:
        return 0
    index.save(args.out)
    size_kb = args.out.stat().st_size / 1024
    print(f"\nwrote {args.out} — {len(index)} spots, {size_kb:.0f} KB")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
