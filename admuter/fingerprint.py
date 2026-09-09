"""Recognise an ad by having heard it before.

Every other voter in this system *infers* — it asks whether a window sounds
like advertising, from loudness, dynamics or words. Inference is what fails on
content mastered as hot as its own ads.

This does not infer. It matches. And the reason it works is measurable in the
annotated corpus: at 12-word granularity, 30.1% of ad audio is a verbatim
repeat of something aired earlier, while 0.0% of content is. Ads repeat.
Television does not. That gap is the entire detector.

WHAT A MATCH BUYS THAT NOTHING ELSE CAN
---------------------------------------
Duration. Every other voter answers "is this window ad audio?" and leaves the
state machine to guess when the break ends via ad_end_windows and a failsafe. A
match instead says "this is spot 4f2a, you are 4.1s into it, it runs 30.0s" --
so the controller can hold the mute for exactly the remaining 25.9s and stop.
For any ad heard before, the whole class of unmuted-too-early and overhang
errors disappears.

It is also the only voter allowed to *start* a mute on its own. The ML voter is
a veto and the transcript voter only extends, because both are guesses and a
wrong mute is the failure that ruins the experience. A fingerprint hit is an
identity, so it carries its own authority.

THE HASH
--------
Haitsma-Kalker sub-fingerprints: per frame, one bit per band from the *double*
difference of band energies -- across bands and across time. The double
difference is the point. It cancels overall level and slow spectral tilt, so
the same spot re-encoded, or played a few dB louder, still hashes identically.
That is what makes it a match rather than a similarity score.

Cold start is per-ad and permanent: the first airing of any spot is always
missed, because there is nothing yet to match it against. The library only ever
helps with airings two onward.
"""

from __future__ import annotations

import json
import logging
import time
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from .config import resolve_path

log = logging.getLogger(__name__)

SAMPLE_RATE = 16000
FRAME_SIZE = 4096          # 256 ms analysis window
HOP_SIZE = 512             # 32 ms hop -> 31.25 frames/sec
BANDS = 33                 # 33 band edges -> 32 bits per frame
BAND_LOW_HZ = 300.0
BAND_HIGH_HZ = 3000.0

# A query needs enough frames to be a claim rather than a coincidence. 96
# frames is ~3.1s, which is also roughly the floor for a confident match.
QUERY_FRAMES = 96
# Haitsma-Kalker's own working threshold. Below this bit error rate two blocks
# are the same audio; above it they are not, with very little in between.
MAX_BIT_ERROR_RATE = 0.35
# An exact 32-bit hash collision between unrelated audio is rare, so a handful
# of frames agreeing on the SAME alignment is already strong evidence. The BER
# check afterwards is what actually decides.
MIN_ALIGNMENT_VOTES = 4
# Silence makes every band energy equal, so every difference is zero and the
# whole frame hashes to 0 -- meaning any two silent stretches match perfectly.
# Ad breaks are full of silent seams, so this is not a corner case: unguarded it
# reports near-perfect matches between unrelated spots. A near-pure tone is
# almost as bad, collapsing to a handful of distinct values.
DEGENERATE_HASHES = frozenset({0, 0xFFFFFFFF})
# A block has to carry enough distinct frames to be an identity rather than a
# texture. Real audio gives close to one distinct hash per frame.
MIN_DISTINCT_HASHES = 32


def is_degenerate(value: int) -> bool:
    return value in DEGENERATE_HASHES


def distinct_informative(hashes: list[int]) -> int:
    return len({h for h in hashes if not is_degenerate(h)})


def _band_edges() -> np.ndarray:
    """Logarithmically spaced band edges, as FFT bin indices."""
    edges = np.logspace(
        np.log10(BAND_LOW_HZ), np.log10(BAND_HIGH_HZ), BANDS + 1
    )
    return np.unique((edges * FRAME_SIZE / SAMPLE_RATE).astype(int))


class Fingerprinter:
    """Audio in, one 32-bit sub-fingerprint per 32 ms out."""

    def __init__(self) -> None:
        self._edges = _band_edges()
        self._window = np.hanning(FRAME_SIZE).astype(np.float32)
        # Carried across calls so a fingerprint taken over two feed() calls is
        # identical to one taken over the concatenation. Streaming and batch
        # must agree, or the library never matches the live audio.
        self._tail = np.zeros(0, dtype=np.float32)
        self._previous_bands: np.ndarray | None = None

    def reset(self) -> None:
        self._tail = np.zeros(0, dtype=np.float32)
        self._previous_bands = None

    def feed(self, audio: np.ndarray) -> list[int]:
        """Hashes for whatever complete frames this audio completes."""
        audio = np.asarray(audio, dtype=np.float32)
        buffer = np.concatenate([self._tail, audio]) if self._tail.size else audio
        hashes: list[int] = []
        offset = 0
        while offset + FRAME_SIZE <= len(buffer):
            frame = buffer[offset:offset + FRAME_SIZE] * self._window
            spectrum = np.abs(np.fft.rfft(frame)) ** 2
            bands = np.array([
                spectrum[self._edges[i]:self._edges[i + 1]].sum()
                for i in range(len(self._edges) - 1)
            ])
            if self._previous_bands is not None:
                # Difference across bands, then across time. Two subtractions,
                # so a constant gain and a constant tilt both cancel.
                across_bands = np.diff(bands)
                previous = np.diff(self._previous_bands)
                bits = (across_bands - previous) > 0
                value = 0
                for i, bit in enumerate(bits[:32]):
                    if bit:
                        value |= 1 << i
                hashes.append(value)
            self._previous_bands = bands
            offset += HOP_SIZE
        self._tail = buffer[offset:]
        return hashes


def bit_error_rate(a: list[int], b: list[int]) -> float:
    """Fraction of differing bits over the overlap. 0.0 is identical audio."""
    n = min(len(a), len(b))
    if n == 0:
        return 1.0
    differing = sum(int(x ^ y).bit_count() for x, y in zip(a[:n], b[:n]))
    return differing / (n * 32)


@dataclass
class Ad:
    """One known spot: what it sounds like and how long it runs."""

    ad_id: str
    hashes: list[int]
    times_seen: int = 1
    first_seen: float = field(default_factory=time.time)
    label: str = ""

    @property
    def duration(self) -> float:
        return len(self.hashes) * HOP_SIZE / SAMPLE_RATE


@dataclass(frozen=True)
class Match:
    ad_id: str
    offset_seconds: float      # how far into the ad the query sits
    remaining_seconds: float   # how much of it is still to play
    bit_error_rate: float
    duration_seconds: float


class FingerprintIndex:
    """Hash -> (ad, frame) postings, queried by alignment vote then verified.

    Exact-hash lookup on its own would be brittle: one flipped bit and a frame
    stops matching. It is not used that way. Lookup only proposes candidate
    *alignments*; agreement between a handful of frames on the same alignment
    is what nominates an ad, and the bit error rate over the whole block is
    what accepts or rejects it.
    """

    def __init__(self) -> None:
        self.ads: dict[str, Ad] = {}
        self._postings: dict[int, list[tuple[str, int]]] = defaultdict(list)

    def __len__(self) -> int:
        return len(self.ads)

    def add(self, ad: Ad) -> None:
        self.ads[ad.ad_id] = ad
        for frame, value in enumerate(ad.hashes):
            # Degenerate frames stay in ad.hashes so alignment arithmetic and
            # the bit-error check still see the real timeline, but they are
            # never allowed to nominate a candidate: silence matching silence
            # is not evidence of anything.
            if not is_degenerate(value):
                self._postings[value].append((ad.ad_id, frame))

    def query(self, hashes: list[int]) -> Match | None:
        if len(hashes) < MIN_ALIGNMENT_VOTES or not self.ads:
            return None
        if distinct_informative(hashes) < MIN_DISTINCT_HASHES:
            # Mostly silence, a tone, or a steady texture. There is nothing here
            # to identify, and matching on it produces confident nonsense.
            return None

        # Vote on (ad, alignment): where in the ad this query would have to
        # start for this frame to line up.
        votes: dict[tuple[str, int], int] = defaultdict(int)
        for position, value in enumerate(hashes):
            for ad_id, frame in self._postings.get(value, ()):
                votes[(ad_id, frame - position)] += 1
        if not votes:
            return None

        best: Match | None = None
        for (ad_id, start), count in sorted(
            votes.items(), key=lambda kv: -kv[1]
        )[:8]:
            if count < MIN_ALIGNMENT_VOTES or start < 0:
                continue
            ad = self.ads[ad_id]
            stored = ad.hashes[start:start + len(hashes)]
            if len(stored) < MIN_ALIGNMENT_VOTES:
                continue
            if distinct_informative(stored) < MIN_DISTINCT_HASHES:
                continue
            ber = bit_error_rate(stored, hashes)
            if ber > MAX_BIT_ERROR_RATE:
                continue
            end_frame = start + len(hashes)
            candidate = Match(
                ad_id=ad_id,
                offset_seconds=end_frame * HOP_SIZE / SAMPLE_RATE,
                remaining_seconds=max(
                    0.0, (len(ad.hashes) - end_frame) * HOP_SIZE / SAMPLE_RATE
                ),
                bit_error_rate=ber,
                duration_seconds=ad.duration,
            )
            if best is None or candidate.bit_error_rate < best.bit_error_rate:
                best = candidate
        return best

    # -- persistence ----------------------------------------------------- #

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps({
            "version": 1,
            "hop_size": HOP_SIZE,
            "sample_rate": SAMPLE_RATE,
            "ads": [
                {"ad_id": a.ad_id, "hashes": a.hashes, "times_seen": a.times_seen,
                 "first_seen": a.first_seen, "label": a.label}
                for a in self.ads.values()
            ],
        }), encoding="utf-8")

    @classmethod
    def load(cls, path: Path) -> "FingerprintIndex":
        index = cls()
        if not Path(path).exists():
            return index
        try:
            payload = json.loads(Path(path).read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            log.warning("could not read ad library %s (%s); starting empty", path, exc)
            return index
        # Hashes are only comparable to audio framed the same way. A library
        # built at different settings is not wrong, it is unusable.
        if (payload.get("hop_size"), payload.get("sample_rate")) != (
            HOP_SIZE, SAMPLE_RATE
        ):
            log.warning("ad library %s was built at different framing; ignoring", path)
            return index
        for entry in payload.get("ads", []):
            index.add(Ad(
                ad_id=entry["ad_id"], hashes=list(entry["hashes"]),
                times_seen=entry.get("times_seen", 1),
                first_seen=entry.get("first_seen", 0.0),
                label=entry.get("label", ""),
            ))
        return index


class RepeatDetector:
    """Finds ads by noticing that something has been heard before.

    This is what makes the library self-building. Content never repeats -- 0.0%
    of the annotated corpus at 12-word granularity -- so audio that recurs at an
    unrelated time is, essentially without exception, an ad. No annotation, no
    labels, no training: the system discovers its own inventory by listening.

    The exception worth knowing is a title sequence, which recurs across every
    episode. It is separated by *when* rather than by sound: an ad recurs at
    arbitrary times, a theme recurs at nearly the same offset into each episode.
    """

    def __init__(
        self,
        min_separation_seconds: float = 90.0,
        history_seconds: float = 3 * 3600.0,
    ) -> None:
        self.min_separation = min_separation_seconds
        self.history_seconds = history_seconds
        self._history: list[tuple[float, list[int]]] = []
        self._index = FingerprintIndex()
        self._counter = 0

    def observe(self, hashes: list[int], timestamp: float) -> str | None:
        """Record a block; return a new ad id if this block is a repeat."""
        if len(hashes) < QUERY_FRAMES:
            return None
        match = self._index.query(hashes)
        self._history.append((timestamp, list(hashes)))
        cutoff = timestamp - self.history_seconds
        while self._history and self._history[0][0] < cutoff:
            self._history.pop(0)

        block_id = f"blk{self._counter:06d}"
        self._counter += 1
        self._index.add(Ad(ad_id=block_id, hashes=list(hashes), first_seen=timestamp))

        if match is None:
            return None
        earlier = self._index.ads[match.ad_id].first_seen
        if timestamp - earlier < self.min_separation:
            # Overlapping windows of the same airing, not a second airing.
            return None
        return match.ad_id


class FingerprintVoter:
    """Runtime matcher: audio in, "this is spot X and it ends at T" out.

    Cheap enough to run inline on the capture thread -- about 31 small FFTs per
    second of audio -- so unlike the transcript voter it needs no thread, no
    queue and no dropped windows. It is also deterministic, which matters
    because offline replay has to reproduce what the live service did.
    """

    def __init__(
        self,
        index: FingerprintIndex,
        max_bit_error_rate: float = MAX_BIT_ERROR_RATE,
        learn: bool = True,
    ) -> None:
        self.index = index
        self.max_bit_error_rate = max_bit_error_rate
        self._fingerprinter = Fingerprinter()
        self._recent: list[int] = []
        self._match: Match | None = None
        self._hold_until: float | None = None
        self._repeats = RepeatDetector() if learn else None
        self.candidates: list[tuple[float, str]] = []

    def reset(self) -> None:
        """Stream restart: the audio is discontinuous, so the buffer is a lie."""
        self._fingerprinter.reset()
        self._recent.clear()
        self._match = None
        self._hold_until = None

    def feed(self, samples: np.ndarray, sample_rate: int, timestamp: float) -> None:
        audio = np.asarray(samples, dtype=np.float32)
        if audio.ndim > 1:
            audio = audio.mean(axis=1)
        factor = max(1, int(round(sample_rate / SAMPLE_RATE)))
        if factor > 1:
            usable = (len(audio) // factor) * factor
            if usable == 0:
                return
            audio = audio[:usable].reshape(-1, factor).mean(axis=1)

        self._recent.extend(self._fingerprinter.feed(audio))
        if len(self._recent) > QUERY_FRAMES * 3:
            self._recent = self._recent[-QUERY_FRAMES * 3:]
        if len(self._recent) < QUERY_FRAMES:
            return

        block = self._recent[-QUERY_FRAMES:]
        match = self.index.query(block)
        if match is not None and match.bit_error_rate <= self.max_bit_error_rate:
            self._match = match
            # The end of this spot, in session time. This is the whole point:
            # no other voter can say when the ad stops, only whether it is one.
            self._hold_until = timestamp + match.remaining_seconds
            log.info(
                "fingerprint hit: %s at %.1fs in (%.1fs left, BER %.3f)",
                match.ad_id, match.offset_seconds, match.remaining_seconds,
                match.bit_error_rate,
            )
        if self._repeats is not None:
            repeat_of = self._repeats.observe(block, timestamp)
            if repeat_of is not None:
                # Not promoted automatically: a repeated 3-second block marks
                # where an ad recurs, not where it begins and ends, and an entry
                # with the wrong boundaries would hold mutes for the wrong span.
                self.candidates.append((timestamp, repeat_of))
                log.info("repeat heard at %.0fs — candidate for the ad library", timestamp)

    def says_ad(self, timestamp: float) -> tuple[bool, float]:
        """(matched, seconds still to run). Nothing matched is (False, 0.0)."""
        if self._hold_until is None or timestamp >= self._hold_until:
            return False, 0.0
        return True, self._hold_until - timestamp

    @property
    def hold_until(self) -> float | None:
        return self._hold_until

    @property
    def last_match(self) -> Match | None:
        return self._match


def load_fingerprint_voter(detection) -> "FingerprintVoter | None":
    """Build the voter a DetectionConfig asks for, or None. Never raises."""
    path = getattr(detection, "fingerprint_library_path", "")
    if not path:
        return None
    try:
        index = FingerprintIndex.load(resolve_path(path))
    except Exception as exc:  # noqa: BLE001 - never take the service down
        level = log.error if detection.fingerprint_enabled else log.warning
        level("ad library unavailable (%s); continuing without it", exc)
        return None
    if len(index) == 0:
        log.warning(
            "ad library %s is empty — fingerprinting can only recognise a spot "
            "it has already heard, so it will do nothing until it is populated "
            "(scripts/build_ad_library.py)", path,
        )
    log.info(
        "ad library loaded from %s (%d spots, %s)",
        path, len(index),
        "voting" if detection.fingerprint_enabled
        else "SHADOW MODE — logged, not counted",
    )
    return FingerprintVoter(index, max_bit_error_rate=detection.fingerprint_max_ber)
