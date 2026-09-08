#!/usr/bin/env python3
"""Mine ad-copy phrases from annotated recordings, by transcribing them.

    python scripts/mine_lexicon.py -i ~/Movies/modern-family-hulu -i ...
    python scripts/mine_lexicon.py --loso ~/Movies --model base.en

The lexicon in ``admuter/transcript.py`` started as informed guesswork. This
replaces guesses with what the advertisers on *this* television actually say:
it transcribes the labelled ad spans and, for contrast, a sample of the
labelled content, then reports the phrases that are common in ads and rare in
the show.

The contrast is the whole point. The most frequent words in ad audio are "the",
"you" and "and", exactly as they are in dialogue, so raw frequency is useless.
What matters is the *ratio* -- a phrase earns its place by being ordinary in
advertising and near-absent from television, which is also precisely the
property that makes it safe to mute on.

Transcription uses faster-whisper rather than Vosk. Vosk is the right choice on
the Pi, where latency rules; this runs offline where accuracy rules, and there
is no macOS Vosk wheel anyway.
"""

from __future__ import annotations

import argparse
import math
import re
import sys
import wave
from collections import Counter
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from admuter.transcript import AD_COPY_PHRASES, normalise  # noqa: E402

from build_dataset import AD_LABELS, DatasetError, pair_chunks, parse_labels  # noqa: E402

# Words too common to ever be evidence, and too common to be worth ranking.
STOPWORDS = {
    "the", "a", "an", "and", "or", "but", "if", "is", "are", "was", "were", "be",
    "been", "to", "of", "in", "on", "at", "for", "with", "that", "this", "it",
    "i", "you", "he", "she", "we", "they", "me", "him", "her", "them", "my",
    "your", "his", "their", "our", "so", "do", "does", "did", "not", "no",
    "yes", "have", "has", "had", "what", "when", "where", "who", "how", "why",
    "all", "just", "like", "get", "got", "know", "think", "going", "gonna",
    "okay", "oh", "yeah", "hey", "well", "now", "then", "there", "here",
}


def read_span(wav: Path, start: float, end: float) -> tuple[np.ndarray, int]:
    """Mono float32 for one labelled span."""
    with wave.open(str(wav), "rb") as handle:
        rate = handle.getframerate()
        channels = handle.getnchannels()
        handle.setpos(min(int(start * rate), handle.getnframes()))
        count = max(0, int((end - start) * rate))
        raw = handle.readframes(min(count, handle.getnframes() - handle.tell()))
    if not raw:
        return np.zeros(0, dtype=np.float32), rate
    audio = np.frombuffer(raw, dtype="<i2").astype(np.float32) / 32768.0
    if channels > 1:
        audio = audio.reshape(-1, channels).mean(axis=1)
    return audio, rate


def resample_16k(audio: np.ndarray, rate: int) -> np.ndarray:
    """Whisper wants 16 kHz. 48/16 is exact, so average whole groups."""
    factor = max(1, int(round(rate / 16000)))
    if factor == 1:
        return audio
    usable = (len(audio) // factor) * factor
    if usable == 0:
        return np.zeros(0, dtype=np.float32)
    return audio[:usable].reshape(-1, factor).mean(axis=1).astype(np.float32)


def spans_for(pairs, content_budget: float) -> tuple[list, list]:
    """(ad spans, content spans) as (wav, start, end), content capped."""
    ads, content = [], []
    used = 0.0
    for wav, label_path in pairs:
        labelled = sorted(parse_labels(label_path), key=lambda s: s.start)
        ad_spans = [s for s in labelled if s.label in AD_LABELS]
        ads.extend((wav, s.start, s.end) for s in ad_spans)
        # Content is everything between the ads, sampled until the budget runs
        # out -- transcribing whole episodes costs hours and buys nothing.
        cursor = 0.0
        for span in ad_spans + [None]:
            stop = span.start if span is not None else cursor + 120.0
            if stop - cursor > 20.0 and used < content_budget:
                take = min(stop - cursor, 90.0, content_budget - used)
                content.append((wav, cursor, cursor + take))
                used += take
            if span is not None:
                cursor = span.end
    return ads, content


def transcribe(model, spans, label: str) -> str:
    out: list[str] = []
    total = sum(e - s for _, s, e in spans)
    done = 0.0
    for wav, start, end in spans:
        audio, rate = read_span(wav, start, end)
        if audio.size == 0:
            continue
        segments, _ = model.transcribe(
            resample_16k(audio, rate), language="en", beam_size=1,
            vad_filter=True, condition_on_previous_text=False,
        )
        out.extend(seg.text for seg in segments)
        done += end - start
        print(f"  {label}: {done:.0f}/{total:.0f}s", end="\r", flush=True)
    print(f"  {label}: {done:.0f}s transcribed" + " " * 20)
    return " ".join(out)


def ngrams(text: str, n: int) -> Counter:
    words = normalise(text).split()
    return Counter(" ".join(words[i:i + n]) for i in range(len(words) - n + 1))


def report(ad_text: str, content_text: str, min_count: int, top: int) -> None:
    ad_words = len(normalise(ad_text).split())
    content_words = len(normalise(content_text).split())
    print(f"\n{ad_words} ad words vs {content_words} content words\n")
    if not ad_words:
        print("nothing transcribed from the ad spans")
        return

    known = {normalise(p) for p in AD_COPY_PHRASES}
    for n in (1, 2, 3, 4):
        ad_counts, content_counts = ngrams(ad_text, n), ngrams(content_text, n)
        scored = []
        for phrase, count in ad_counts.items():
            if count < min_count:
                continue
            if n == 1 and phrase in STOPWORDS:
                continue
            # Rate per 10k words each side, with add-one smoothing so a phrase
            # absent from content does not divide by zero -- and so a phrase
            # seen once is not ranked above one seen fifty times.
            ad_rate = 1e4 * count / ad_words
            content_rate = 1e4 * (content_counts.get(phrase, 0) + 1) / max(
                content_words + 1, 1
            )
            scored.append((math.log2(ad_rate / content_rate), count,
                           content_counts.get(phrase, 0), phrase))
        scored.sort(reverse=True)
        print(f"--- {n}-gram: most ad-distinctive (log2 ratio, ad n, content n) ---")
        for ratio, ad_n, content_n, phrase in scored[:top]:
            mark = "  [already in lexicon]" if phrase in known else ""
            print(f"  {ratio:+6.2f}  {ad_n:>4} / {content_n:<4}  {phrase}{mark}")
        print()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("-i", "--indir", type=Path, action="append", dest="indirs",
                        metavar="DIR")
    parser.add_argument("--loso", type=Path, metavar="PARENT",
                        help="use every session subdirectory of PARENT")
    parser.add_argument("--model", default="base.en",
                        help="faster-whisper model (tiny.en, base.en, small.en)")
    parser.add_argument("--content-budget", type=float, default=2400.0,
                        help="seconds of content to transcribe for contrast")
    parser.add_argument("--min-count", type=int, default=3)
    parser.add_argument("--top", type=int, default=30)
    parser.add_argument("--dump", type=Path, help="write raw transcripts here")
    args = parser.parse_args(argv)

    dirs = list(args.indirs or [])
    if args.loso:
        dirs += [d for d in sorted(args.loso.iterdir())
                 if d.is_dir() and any(d.glob("*.wav"))]
    if not dirs:
        print("error: give -i or --loso", file=sys.stderr)
        return 2

    ad_spans, content_spans = [], []
    for d in dirs:
        try:
            pairs = pair_chunks(d)
        except DatasetError as exc:
            print(f"warning: skipping {d.name} — {exc}", file=sys.stderr)
            continue
        ads, content = spans_for(pairs, args.content_budget / len(dirs))
        ad_spans += ads
        content_spans += content
    print(f"{len(ad_spans)} ad spans, {len(content_spans)} content samples "
          f"from {len(dirs)} sessions")

    from faster_whisper import WhisperModel

    model = WhisperModel(args.model, device="cpu", compute_type="int8")
    ad_text = transcribe(model, ad_spans, "ads")
    content_text = transcribe(model, content_spans, "content")

    if args.dump:
        args.dump.write_text(
            f"=== ADS ===\n{ad_text}\n\n=== CONTENT ===\n{content_text}\n",
            encoding="utf-8",
        )
        print(f"wrote {args.dump}")

    report(ad_text, content_text, args.min_count, args.top)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
