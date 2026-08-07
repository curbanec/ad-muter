#!/usr/bin/env python3
"""Record N seconds from the SPDIF capture device to a labelled WAV.

Run this on the Pi while the TV is playing, and hit it right before an ad break:

    python scripts/record_sample.py --seconds 180 --label mixed --note "S1E4 ad break"
    python scripts/record_sample.py --seconds 60 --label ad
    python scripts/record_sample.py --seconds 60 --label content

Files land in samples/ as ``<label>_<timestamp>.wav`` with a matching ``.json``
sidecar. Replay them with scripts/replay_wav.py to tune thresholds; keep them
around as the seed corpus for the Phase 2 classifier.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import sys
import wave
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from admuter.capture import resolve_device  # noqa: E402
from admuter.config import Config, ConfigError  # noqa: E402

DEFAULT_CONFIG = Path(__file__).resolve().parent.parent / "config.yaml"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("-s", "--seconds", type=float, default=60.0)
    parser.add_argument(
        "-l",
        "--label",
        default="mixed",
        help="ad | content | mixed | anything else you want to grep for later",
    )
    parser.add_argument("-o", "--outdir", type=Path, default=Path("samples"))
    parser.add_argument("-c", "--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("-d", "--device", help="override audio.device from config")
    parser.add_argument("-n", "--note", default="", help="free-text note for the sidecar")
    args = parser.parse_args(argv)

    try:
        config = Config.load(args.config)
    except ConfigError as exc:
        print(f"config error: {exc}", file=sys.stderr)
        return 2

    try:
        import sounddevice as sd
    except (ImportError, OSError) as exc:
        print(f"sounddevice unavailable: {exc}", file=sys.stderr)
        return 1

    spec = args.device or config.audio.device
    sample_rate = config.audio.sample_rate
    channels = config.audio.channels
    frames = int(round(sample_rate * args.seconds))

    try:
        device = resolve_device(spec, sd.query_devices)
    except Exception as exc:
        print(f"cannot open {spec!r}: {exc}", file=sys.stderr)
        return 1

    args.outdir.mkdir(parents=True, exist_ok=True)
    stamp = dt.datetime.now().strftime("%Y%m%dT%H%M%S")
    wav_path = args.outdir / f"{args.label}_{stamp}.wav"

    print(
        f"recording {args.seconds:.0f}s from {spec} "
        f"({sample_rate} Hz, {channels} ch) -> {wav_path}"
    )
    try:
        data = sd.rec(
            frames,
            samplerate=sample_rate,
            channels=channels,
            dtype="int16",
            device=device,
        )
        sd.wait()
    except KeyboardInterrupt:
        print("\ninterrupted — nothing written", file=sys.stderr)
        return 130
    except Exception as exc:
        print(f"capture failed: {exc}", file=sys.stderr)
        return 1

    with wave.open(str(wav_path), "wb") as wav:
        wav.setnchannels(channels)
        wav.setsampwidth(2)
        wav.setframerate(sample_rate)
        wav.writeframes(data.tobytes())

    sidecar = wav_path.with_suffix(".json")
    sidecar.write_text(
        json.dumps(
            {
                "label": args.label,
                "note": args.note,
                "recorded_at": dt.datetime.now().astimezone().isoformat(),
                "device": spec,
                "sample_rate": sample_rate,
                "channels": channels,
                "seconds": args.seconds,
                "wav": wav_path.name,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    print(f"wrote {wav_path} ({wav_path.stat().st_size / 1e6:.1f} MB) and {sidecar.name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
