# admuter — Netflix ad muter (Phase 1)

Listens to the TV's optical output, notices when Netflix cuts to an ad, and mutes
the TV over Roku ECP until the show comes back.

Netflix stitches its ads server-side: the Roku reports the same playback state
during an ad as during the show, so `query/media-player` is useless here. The
audio, however, gives it away — there is a short near-silent seam at the join,
and ad audio is mastered louder and much more compressed than show audio. Phase 1
detects exactly that, with rule-based heuristics and no ML.

```
SPDIF ──> USB capture ──> 1s windows ──> features ──> detector ──> controller ──> Roku ECP
          (sounddevice)   (numpy)        RMS/crest/  AD_STARTED   CONTENT →      VolumeMute
                                         centroid/   AD_ENDED     AD_SUSPECTED
                                         silence                  → MUTED
```

**Latency**: an ad is typically muted 1–2 s after it starts (one window to detect,
one or more to confirm). That is the accepted budget for Phase 1.

**Bias**: when the evidence is ambiguous the system does nothing. A missed ad is
annoying; muting real dialogue is worse.

**The detector is a binary classifier** — a function from one second of audio
features to "ad" or "not ad", plus a confidence. Phase 1 writes that function by
hand; Phase 2 trains it. See [The model](#the-model) for what the features
measure and why the decision boundary sits where it does.

## Hardware

| Piece | What's confirmed working |
| --- | --- |
| Host | Raspberry Pi 5 (1 GB), Raspberry Pi OS Bookworm 64-bit, hostname `admuter`, user `chris`, wired ethernet |
| Capture | Cubilux USB SPDIF receiver (USB ID `0c76:1170`) → ALSA card `Receiver`, `plughw:CARD=Receiver,DEV=0`, 48 kHz / 2 ch / S16_LE |
| TV | Hisense Roku TV at `192.168.0.12`, ECP on port 8060 |

Sanity-check the capture chain before touching this repo:

```bash
arecord -D plughw:CARD=Receiver,DEV=0 -f S16_LE -r 48000 -c 2 -d 5 /tmp/test.wav
curl -s http://192.168.0.12:8060/query/device-info | head
```

## Install

Bookworm's Python is externally managed, so use a venv (recommended) — PortAudio
is a system package either way:

```bash
sudo apt update
sudo apt install -y python3-venv libportaudio2

git clone <this repo> ~/ad-muter
cd ~/ad-muter
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt        # add -r requirements-dev.txt for pytest
```

If you would rather not use a venv, `pip install --break-system-packages -r
requirements.txt` works, but then the systemd unit must call `/usr/bin/python3`
instead of `.venv/bin/python`.

Dependencies are `sounddevice`, `numpy`, `requests`, `PyYAML` — nothing else.
Python ≥ 3.11.

One of those four is only half a pip dependency. `sounddevice` is a `cffi`
binding, not an implementation: it needs the **PortAudio system library**, which
pip will not install for you. That is what `libportaudio2` in the `apt` line
above is for. Skip it and `pip install sounddevice` still succeeds — the failure
arrives later, at the first import, as `OSError: PortAudio library not found`.
`admuter` catches that and re-raises it with the fix attached:

```
sounddevice/PortAudio is unavailable. On Raspberry Pi OS:
sudo apt install libportaudio2 && pip install sounddevice
```

## Configure

Everything tunable lives in `config.yaml`; the code has no hidden thresholds.
Unknown keys are rejected at startup, so a typo fails loudly instead of silently
keeping the default.

The values you are most likely to change first:

```yaml
audio:
  device: "plughw:CARD=Receiver,DEV=0"   # `arecord -L` lists these
roku:
  host: "192.168.0.12"
  netflix_only: true                     # only arm detection while Netflix is on screen
```

Validate a config without running anything:

```bash
.venv/bin/python -m admuter --print-config
```

## Run it manually

Run from the repo directory — the package is imported from there, not
pip-installed (the systemd unit sets `WorkingDirectory` and `PYTHONPATH` for the
same reason).

Start in dry-run — it detects and logs, but never sends a keypress:

```bash
.venv/bin/python -m admuter --dry-run --log-level DEBUG
```

Then for real:

```bash
.venv/bin/python -m admuter
```

Useful flags: `--config PATH`, `--log-level {DEBUG,INFO,…}`, `--dry-run`,
`--print-config`, `--version`. Ctrl-C (or SIGTERM) shuts down cleanly and
unmutes the TV first if we were the ones who muted it.

What the log looks like when it works:

```
INFO  admuter.controller  AD_SUSPECTED (1/1) conf=0.50 — gap=0.50s then loudness +6.1dB / crest -0.4dB vs baseline
INFO  admuter.controller  MUTE (1/1 windows) — gap=0.50s then loudness +6.1dB / crest -0.4dB vs baseline
INFO  admuter.controller  UNMUTE — 6 non-ad windows after 78s
```

Per-window feature lines are DEBUG. Steady-state `NO_CHANGE` decisions are
deduplicated to DEBUG with an INFO heartbeat every `decision_heartbeat_windows`
windows, so the journal does not grow by one line per second forever; set
`logging.verbose_decisions: true` if you want literally every decision at INFO.

## First run on the Pi

The capture path is the one part of this system that has never run. All hardware
testing so far went through `arecord`; the service reaches the card through
`sounddevice`/PortAudio instead, which is a different layer with its own device
naming. `arecord -l` listing the card is *not* evidence that PortAudio can open
it.

Work through these in order. Each step adds exactly one component, so whichever
one fails tells you where the problem is.

**1. Can PortAudio see the card?**

```bash
.venv/bin/python -m sounddevice
```

Expect the receiver in the list with a non-zero input count:

```
  1 Receiver: USB Audio (hw:1,0), ALSA (2 in, 0 out)
```

An empty list, or an `OSError` about a missing PortAudio library, means
`libportaudio2` is not installed — see [Install](#install). This is a system
package; pip cannot supply it.

**2. Does the config load?**

```bash
.venv/bin/python -m admuter --print-config
```

Touches no hardware: it parses `config.yaml`, validates it, prints the resolved
`Config`, and exits 0. Exit code 2 with `config error: …` means a typo or an
unknown key. Run this after every config edit — it is the cheapest way to catch
a mistake that would otherwise crash the service at start.

**3. Does the whole chain run, without touching the TV?**

```bash
.venv/bin/python -m admuter --dry-run --log-level DEBUG
```

This is the first step that opens the sound card and talks to the Roku. Three
lines confirm the chain is up:

```
INFO  admuter.capture  audio device 'plughw:CARD=Receiver,DEV=0' resolved to 'Receiver: USB Audio (hw:1,0)'
INFO  admuter          connected to Living Room TV (43S455, serial X00000000000)
INFO  admuter.capture  capturing from plughw:CARD=Receiver,DEV=0 at 48000 Hz, 2 ch, 1.00s windows
```

Then one DEBUG line per window:

```
DEBUG admuter.controller win 3 t=3.0 rms=-21.9 dBFS peak=-9.9 crest=12.0 dB centroid=1466 Hz silence=0% gap=0.00s state=CONTENT
```

Play something with dialogue and check the numbers are plausible before trusting
any of it. The middle column is the 10th–90th percentile actually measured over
the 6881-window jungle-movie recording, through this same optical chain — useful
as a sanity range, not as a specification:

| Field | Measured p10–p90 | Suspicious |
| --- | --- | --- |
| `rms` | −40 to −18 dBFS (median −28), moving with the mix | pinned at `-120.0` — the stream is open but no audio is arriving |
| `crest` | 12 to 19 dB (median 14) | stuck near 0 dB regardless of content |
| `centroid` | 1600 to 4000 Hz (median 2500) | `0` Hz, which only happens for a silent buffer |
| `silence` | 0% through speech and music | 100% constantly |

The one that matters is `rms`. Pinned at `-120.0` while the movie plays means the
capture opened something that is not carrying the TV's audio — check the optical
output is enabled and set to **PCM**, since everything downstream assumes decoded
PCM samples. Values that move sensibly with the mix mean the chain works end to
end, which is the whole point of this step.

### What device resolution failure looks like

`resolve_device()` tries the configured string first, then — for a
`plughw:CARD=…` spec — the bare card name, which sounddevice resolves by
case-insensitive substring match. The fallback is the untested path, so it is
worth knowing its output on sight.

On success it logs which candidate won, and the resolved name tells you whether
the fallback was used:

```
INFO  admuter.capture  audio device 'plughw:CARD=Receiver,DEV=0' resolved to 'Receiver: USB Audio (hw:1,0)'
```

If **neither** candidate matches, `resolve_device()` raises `CaptureError`
listing both attempts:

```
no input device matched 'plughw:CARD=Receiver,DEV=0' ('plughw:CARD=Receiver,DEV=0':
No input device matching 'plughw:CARD=Receiver,DEV=0'; 'Receiver': No input device
matching 'Receiver'). Run `python -m sounddevice` to list what PortAudio can see.
```

**The service does not stop when that happens**, and this is the part most
likely to mislead you. `AudioCapture.windows()` treats it as a temporary
condition — a vanished device is normal when the TV is off — so it logs a
warning, backs off, and retries forever, doubling the delay from 1 s up to
`audio.retry_max_seconds` (30 s):

```
WARNING admuter.capture  audio capture unavailable (no input device matched
'plughw:CARD=Receiver,DEV=0' (…)); retrying in 1.0s (is the TV on?)
```

The `(is the TV on?)` suffix is generic to that retry path. A misconfigured
`audio.device` produces the same line as a genuinely absent card, so read the
message body, not the suffix. Repeating warnings with a growing backoff and no
`capturing from …` line ever appearing means the device name is wrong — fix
`audio.device` using the names from step 1.

One more variant worth recognising: if the bare card name matches *more than one*
input, sounddevice raises `Multiple input devices found for 'Receiver': …` with
the candidates listed. Fix that by putting the specific name or the device index
from step 1 into `audio.device` — a plain integer is accepted as an index.

### The first run that can actually mute

Only after the three checks above pass, and in this order:

1. **`--dry-run --log-level DEBUG`, through a real ad break.** Mute and unmute
   decisions are logged, never sent, so a mistuned detector cannot mute the TV
   in the middle of a movie. This is the only step where a wrong threshold is
   free. Confirm the feature lines look sane, then watch what it *would* have
   done across an actual break — each suppressed keypress logs as:

   ```
   INFO  admuter.roku  [dry-run] would send keypress VolumeMute
   ```

   One at the start of the break and one at the end is the shape you want.
2. **Consider `roku.netflix_only: false` for the first live test.** With gating
   on, the detector stays disarmed unless `query/active-app` reports Netflix —
   a second, silent failure mode that looks exactly like a detection failure. If
   nothing ever fires and the log shows no `Netflix active — detection armed`
   line, you are debugging ECP, not the detector. Turning gating off for one
   session removes that variable; turn it back on afterwards.
3. **Install the systemd unit last.** Only once a manual run has muted and
   unmuted correctly across a real ad break. Under systemd the process restarts
   on failure and logs to the journal, which makes exactly this class of
   first-run problem harder to watch. See
   [Deploy as a systemd service](#deploy-as-a-systemd-service).

Note that `--dry-run` still opens the capture device and still queries the Roku
for device info and active app — the only thing it suppresses is the keypress.
That is what makes it a complete rehearsal rather than a partial one.

## Tests

The suite runs anywhere — no Pi, no capture device, no TV. HTTP is mocked and the
audio is synthetic numpy.

```bash
.venv/bin/pip install -r requirements-dev.txt
.venv/bin/python -m pytest tests/ -q
```

`tests/test_features.py` covers the DSP, `tests/test_detector.py` the heuristics,
`tests/test_roku.py` the ECP client (including the toggle-state bookkeeping), and
`tests/test_controller.py` the state machine, failsafes, and Netflix gating.

## Deploy as a systemd service

The unit assumes the repo at `/home/chris/ad-muter` with its venv at
`.venv`. Edit `systemd/admuter.service` if your paths differ.

```bash
sudo cp systemd/admuter.service /etc/systemd/system/admuter.service
sudo systemctl daemon-reload
sudo systemctl enable --now admuter.service

systemctl status admuter.service
journalctl -u admuter.service -f          # live logs
journalctl -u admuter.service -p warning  # just the interesting bits
```

After editing `config.yaml`: `sudo systemctl restart admuter.service`.

The service runs as `chris` with the `audio` supplementary group, restarts on
failure after 5 s, and gets a SIGTERM on stop so it can unmute before exiting.

## Recording and replaying samples

This is the loop you use to tune thresholds — and the samples double as the seed
corpus for the Phase 2 classifier.

**Record on the Pi**, ideally starting just before an ad break:

```bash
.venv/bin/python scripts/record_sample.py --seconds 180 --label mixed --note "S1E4 ad break"
.venv/bin/python scripts/record_sample.py --seconds 60 --label ad
.venv/bin/python scripts/record_sample.py --seconds 60 --label content
```

Files land in `samples/` as `<label>_<timestamp>.wav` with a JSON sidecar
(label, note, device, sample rate, duration). One minute of 48 kHz stereo is
about 11 MB.

**Replay anywhere** — same feature extractor, same detector, same state machine,
with the TV replaced by a stub:

```bash
.venv/bin/python scripts/replay_wav.py samples/mixed_20260806T203102.wav
.venv/bin/python scripts/replay_wav.py samples/*.wav --only-changes
.venv/bin/python scripts/replay_wav.py sample.wav --set detection.ad_crest_delta_db=0.5
.venv/bin/python scripts/replay_wav.py sample.wav -o /tmp/rows.csv
```

Output is one row per window plus a summary of the mute spans it would have
produced:

```
    time     rms    peak  crest  centroid  sil%   gap      event  conf         state  mute  reason
 00:39.0   -27.5   -13.2   14.3     11979     0  0.00  NO_CHANGE  0.64       CONTENT     -  content
 00:40.0    -9.6    -5.2    4.4     12005    50  0.50 AD_STARTED  1.00  AD_SUSPECTED     -  gap=0.50s then loudness +18.9dB / crest -11.2dB vs baseline
 00:41.0    -6.6    -5.2    1.4     11985     0  0.00  NO_CHANGE  1.00         MUTED MUTED  ad continues (1s)

mute spans that would have been applied:
  00:41.0 -> 01:12.0 (31s)
  total muted: 31s of 96s (32%)
```

`--set section.key=value` overrides config for one run, so you can sweep a
threshold without editing the file. Once a value works across all your samples,
put it in `config.yaml`.

**Scoring against labels.** Once a recording has Audacity label files alongside
the WAVs, `score_detector.py` replaces eyeballing replay output with three
numbers — breaks caught, mute latency, and seconds of content falsely muted:

```bash
.venv/bin/python scripts/score_detector.py -i samples/movie
.venv/bin/python scripts/score_detector.py -i samples/movie --sweep controller.confirm_windows=1,2,3
```

Window-level accuracy is the wrong lens for this system — the controller's
confirmation and hysteresis mean isolated bad windows never reach the TV, and
[the tuning guide](#what-one-labelled-recording-actually-showed) has a worked
example of a change that improves per-window AUC while making the TV behave
worse. Score the mute spans, not the windows.

**Collecting training data in production**: set `logging.feature_log_enabled:
true` and every window is appended to `logs/features.jsonl` (or `.csv`) with its
features, the detector's metrics, the decision, and the state. Annotate the ad
spans later and you have a labelled set for Phase 2.

## The model

Strip away the plumbing and what's left is a **binary classifier**: a function
that takes the audio features of a one-second window and outputs a guess — "ad"
or "not ad" — with a confidence number attached. That's it. That's the whole
model. Phase 1 implements that function with hand-written rules; Phase 2 will
implement it with a trained one. Everything else in this repo — capture,
confirmation, mute bookkeeping, failsafes — exists to feed that function and to
survive its mistakes.

### The two outputs, and why they're different

`Decision` (`admuter/detector.py`) carries the classifier's answer twice, in two
forms:

* `ad_profile` + `confidence` — the **level**. The classifier's actual output for
  this window: does this second of audio sound like an ad? True on *every*
  ad-like window.
* `event` — the **edge**. `AD_STARTED` / `AD_ENDED` / `NO_CHANGE`: the moment the
  answer flips. Derived from the level plus temporal state.

That split is the standard architecture for audio event detection: a frame-level
classifier, then a decoding layer that turns a noisy per-frame sequence into
clean segment boundaries. The controller consumes both — the edge opens and
closes the state machine, the level is what it counts to confirm. A Phase 2
model only has to produce the level honestly; the edge falls out of it.

### What the features actually measure

Four physical quantities, each a proxy for something about how the audio was
mastered:

| Feature | Physical meaning | Why an ad differs |
| --- | --- | --- |
| **RMS (dBFS)** | Integrated loudness | Ads are mastered hot — but streaming platforms loudness-normalize, so on its own this is a *weak* signal. Weak, and yet still the strongest of the four in practice |
| **Crest factor** (peak ÷ RMS, in dB) | Dynamic range | Expected to be the money feature. On the one recording measured so far it is nearly inert — see below |
| **Spectral centroid** | Brightness: the first moment of the magnitude spectrum | Ad beds are music- and voiceover-heavy, so brighter than dialogue-dominant show audio |
| **Silence structure** | Where the near-silent frames sit within the window | Server-side ad insertion concatenates separately encoded segments, and the join usually leaves a brief digital-silence seam |

**Why crest factor was supposed to carry most of the weight.** Loudness
normalization is exactly what should make it work. Under a fixed
integrated-loudness target, the only way to make something *sound* louder is to
raise its average level without exceeding the peak ceiling — which means
compressing and limiting the dynamic range. Advertising audio is mastered that
way as a matter of course. So the normalization that neuters raw loudness as a
feature ought to push ads into a distinctive crest-factor regime: peak and RMS
collapse toward each other, while dialogue keeps them far apart.

**On the first labelled recording, it doesn't.** Per-window crest factor
separates ads from content at AUC 0.501 — a coin flip. Median `crest_db` is
14.14 for ad windows and 14.15 for content windows. Whatever the mastering
argument predicts, these ads are not measurably more compressed than this
movie's audio at a 1-second window size, and an `ad_crest_delta_db` of 2.0 dB
vetoed nearly every real ad window: it alone cost 2 of the 3 breaks.

That does not mean the feature is worthless. Reduced to a *sign* test
(`ad_crest_delta_db: 0.0` — "no more dynamic than the baseline") it is what
ends mutes on time; removing it entirely raises ad coverage from 51% to 87% but
adds 43 s of false mute, because the ad state then overhangs into returning
dialogue. It earns its place as a weak veto, not as the primary discriminator.

**What actually carries the detector is the transition cue.** Near-silent seams
are strongly ad-associated: 7.3% of ad windows contain a ≥0.2 s silent run
versus 0.3% of content windows. Loudness alone gives a window-level precision of
0.10; the same loudness test gated on a recent gap gives 0.75. The cue is not a
refinement on top of the profile — it is the reason the profile is usable at
all, which is why `require_transition_cue` should stay `true`.

### Why every threshold is relative

None of these features has a meaningful absolute value. What counts as "loud"
depends on the TV volume, the mix, the show, the genre — a nature documentary
and an action series don't share a scale. What *does* generalize is the contrast
between the current window and the recent past, so the detector maintains an
exponential moving average of content features (`baseline_alpha: 0.05`, a ~20 s
time constant) and classifies on the *deltas*. Same idea as an adaptive noise
floor in a voice-activity detector: estimate the background, then look for
departures from it. This is also why the detector refuses to answer at all until
`baseline_min_windows` have gone by — a classifier with no reference frame is
guessing.

### The operating point is deliberately lopsided

A binary classifier makes two kinds of mistake, and here they cost wildly
different amounts. A false negative is a missed ad: mildly annoying. A false
positive mutes real dialogue: it ruins the thing you're trying to watch. With a
loss function that asymmetric, the right operating point is nowhere near the
"balanced" one — high precision, and recall pays for it. Four mechanisms buy
that precision:

1. **Conjunction, not disjunction.** The ad profile requires louder **and** less
   dynamic than baseline. Intersecting two conditions instead of accepting either
   shrinks the positive region considerably — though see above: on the recording
   measured so far the crest half of that conjunction contributes far less than
   the loudness half, and set too tight it simply vetoes everything.
2. **A prior on timing.** The transition cue means the classifier's answer is
   only acted on shortly after a silent seam — the only place an ad can actually
   begin. Ad-like audio in the middle of a scene is ignored by construction.
3. **Temporal integration.** `confirm_windows` requires consecutive positives.
   Isolated flukes — one loud compressed second — die here; sustained ad audio
   sails through. (Errors aren't independent, so this isn't literally p^N, but it
   kills the single-window failure mode that dominates in practice.)
4. **Hysteresis.** Entering the ad state and leaving it use different criteria,
   so the system can't chatter around the boundary the way a single threshold
   would. Three things differ between the two directions: `min_ad_seconds` and
   `ad_end_windows` govern *when* the state may end, and
   `ad_stay_loudness_delta_db` sets a lower loudness bar for *staying* in an ad
   than `ad_loudness_delta_db` demands for entering one — so a quiet spot
   mid-break does not read as content.

Every knob in the tuning guide below moves this operating point. That's all
tuning is: sliding the decision boundary along the precision/recall trade-off,
in a system where one of those two errors is much more expensive than the other.

### What Phase 2 changes

Only the function. A trained model sees the same windows, outputs the same
`Decision`, and inherits the same confirmation, hysteresis, and failsafes. What
it gains is the ability to learn feature interactions that hand-written rules
can't express — and to output a *calibrated* confidence, so the asymmetric loss
above can be applied explicitly as a probability threshold rather than
implicitly through conjunctions and counters.

## Tuning guide

Two failure modes, opposite fixes. Change **one knob at a time** and re-run
`replay_wav.py` over your saved samples — that is much faster than waiting for
the next real ad break. If you have labelled chunks, `score_detector.py` is
better still: it runs the real detector and controller over them and scores the
mute spans that come out.

```bash
.venv/bin/python scripts/score_detector.py -i samples/movie
.venv/bin/python scripts/score_detector.py -i samples/movie \
    --sweep detection.ad_end_windows=2,4,6,8
```

### What one labelled recording actually showed

> **These numbers are provisional.** They come from a single sitting: twelve
> 10-minute chunks of one movie on one service, three ad breaks, 396 ad windows
> out of 6881. One title, one mix, one night. Several findings below directly
> contradict what this README used to advise, which is reason to trust
> measurement over theory — but three breaks is not a sample, and nothing here
> has been reproduced on a second recording. The values in `config.yaml` are the
> best available guess, not settled defaults. Re-run the sweeps on your own
> labelled chunks before believing any of it.

Scoring the shipped config against that recording, before and after tuning:

| | stock | tuned |
| --- | --- | --- |
| breaks caught | 1 / 3 | **3 / 3** |
| content falsely muted | 0 s | **0 s** |
| ad audio actually muted | 10 s (3%) | **203 s (51%)** |
| worst mute latency | +143 s | **+2.8 s** |

Three things fell out of the sweeps, in descending order of how much they
mattered:

1. **Crest factor did not discriminate** (AUC 0.501), and the old default of
   2.0 dB was a veto on nearly every real ad window. Relaxing it to a sign test
   at `0.0` is the single largest improvement in the table above. See
   [What the features actually measure](#what-the-features-actually-measure).
2. **`ad_end_windows: 2` was unmuting mid-break.** These breaks are several
   spots with silent seams between them, and the detector read each seam as
   content. Raising it to 6 took ad coverage from 24% to 51%. Coverage well
   under 100% on a *caught* break is the signature of this, and it is what to
   look at before touching anything that affects whether a break is caught.
3. **A slower baseline made the system worse, not better** — the opposite of
   what the per-window numbers predict. Offline, `delta_rms` AUC improves
   monotonically as `baseline_alpha` drops (0.574 at 0.05 → 0.673 at 0.005). But
   end-to-end, lower alpha leaves the baseline stale under a loud stretch of
   content and fires spurious mutes: 0 s of false mute at 0.05, 23 s at 0.02,
   61 s at 0.005. Better window-level separation, worse behaviour on the TV.
   This is why window-level AUC is the wrong thing to optimise here, and why
   `baseline_alpha` was left at 0.05.

Two things the tuning could **not** fix from config alone:

* **Coverage stops at ~51% if you refuse to trade away false mutes.** The
  remaining ad audio is in the middle of a break — mutes end at an inter-spot
  seam and the next spot never re-fires the cue. Every setting that closed that
  gap (`ad_end_windows: 8+`, `ad_crest_delta_db: -2` or lower,
  `ad_loudness_delta_db: 2.5` or lower) bought coverage with overhang into
  returning dialogue. Holding the mute *through* a seam and re-acquiring *after*
  one are the same knob, and this threshold set cannot separate them.
* **Mute latency on one break is +2.8 s, over the 1–2 s budget.** Of ~1,200
  configurations searched, none caught all three breaks with zero false mutes
  and stayed at or under 2 s. That break's audio simply takes ~3 windows to look
  ad-like. Latency is the last of the three priorities, so it lost.

### Failure mode 1: it mutes real content (the serious one)

A loud, compressed scene right after a quiet one can look like an ad. First work
out *which* false mute you have, because they have different causes.
`score_detector.py` prints a `spurious mute` line for a mute that never touched a
break; anything else in the false-mute total is **overhang** — a real break's
mute running past the end of the break into returning dialogue. On this
recording overhang was by far the larger share.

For overhang past the end of a break:

| Knob | Direction | Effect |
| --- | --- | --- |
| `detection.ad_end_windows` | ↓ 6 → 5 | Ends the ad sooner after the profile lapses. The main overhang control; 8 and above ran ~18 s long, 10 ran ~53 s long. |
| `detection.ad_stay_loudness_delta_db` | ↑ toward `ad_loudness_delta_db` | Raises the bar for *staying* in an ad, so the state lapses sooner once the audio stops looking ad-like. Setting it equal to `ad_loudness_delta_db` disables the hysteresis entirely and restores the single-threshold behaviour. |
| `detection.ad_crest_delta_db` | ↑ 0 → 0.5–1 | Tightens the veto that ends mutes. Costs coverage quickly (51% → 30% at 1.0), but costs no false mutes. |

For a spurious mute in the middle of content:

| Knob | Direction | Effect |
| --- | --- | --- |
| `detection.ad_loudness_delta_db` | ↑ 3 → 3.5–4 | Demands a bigger loudness jump over the baseline. Now the primary discriminator, so this is the first thing to reach for. Below 3.0 false mutes reappeared. |
| `controller.confirm_windows` | ↑ 1 → 2 | Requires the ad profile to hold for another second. Historically the cheapest safe fix, and still the first thing to undo if the tuned config misbehaves — but on this recording it cost a whole break, so check what it takes with you. |
| `detection.require_transition_cue` | keep `true` | With this off, loudness alone can fire — precision 0.10 versus 0.75. Do not turn this off. |
| `detection.max_gap_seconds` | ↓ 1.5 → 0.8 | Long pauses in a quiet scene stop counting as seams. |
| `detection.baseline_alpha` | ↑ 0.05 → 0.07 | **Note the direction** — this reverses earlier advice in this file. A *faster* baseline tracks a loud stretch of content and stops calling it an ad. Slowing it down measurably increased false mutes. |

If a false mute does slip through, `detection.max_ad_seconds` and
`controller.max_mute_seconds` bound the damage — with the shipped values the TV
can never stay muted for more than ~210 s. Do not set `max_ad_seconds` below the
longest ad break you expect: at the old default of 120 s the failsafe would have
unmuted 37 s before the end of the 158 s break in this recording.

### Failure mode 2: it misses ads

Check the coverage percentage first. Missing a break entirely and covering only
part of one are different problems with different knobs — if breaks are being
caught but coverage is low, you want `ad_end_windows`, not the profile
thresholds.

| Knob | Direction | Effect |
| --- | --- | --- |
| `detection.ad_end_windows` | ↑ 6 → 8 | **For low coverage on caught breaks.** Holds the ad state across the silent seams between spots instead of unmuting at each one. Took coverage from 24% to 51% here; watch for overhang as you raise it. |
| `detection.ad_stay_loudness_delta_db` | ↓ 1 → 0 or below | **For a break that unmutes during a soft spot.** Only the *stay* bar drops, so it holds an ad through a quiet passage without making it any easier to enter one by mistake. Note this did not move coverage on the one recording measured — the gaps there were between spots, not inside them. |
| `detection.ad_crest_delta_db` | ↓ 0 → -1 | Accepts ads that are no less dynamic than the show — which, on the recording measured, is most of them. Below about -2 this stops vetoing anything at all. |
| `detection.ad_loudness_delta_db` | ↓ 3 → 2.5 | Netflix normalizes loudness fairly well, so this delta can be genuinely small. |
| `controller.confirm_windows` | ↓ 2 → 1 | Mutes on the first ad-looking window. Faster, noticeably riskier — but it was the only way to catch the third break here. |
| `detection.loudness_jump_db` | ↓ 3 → 2 | Arms the cue on a subtler shift across the seam. |
| `detection.min_gap_seconds` | ↓ 0.2 → 0.1 | Catches shorter seams. Also raises the false-cue rate. |
| `detection.cue_grace_seconds` | ↑ 3 → 5 | Gives the ad profile longer to become obvious after the seam. |
| `detection.silence_dbfs` | ↑ -60 → -50 | Use if the seam is not digital silence but a low noise floor — check the `sil%` and `gap` columns in a replay: if a seam you can hear shows `gap 0.00`, this is the knob. |
| `detection.require_transition_cue` | `false` | Last resort: the ad profile alone can fire. Expect more false mutes. |

### Diagnosing from a replay

* `gap` stays `0.00` at a seam you can hear → raise `silence_dbfs` (the seam is
  not silent enough to count) or lower `min_gap_seconds` (it is too short).
* `gap` is right but no `AD_STARTED` → look at `crest` and `rms` versus the
  `baseline_*` metrics in the `-o` dump; whichever delta falls short of its
  threshold is the one to relax.
* `AD_STARTED` fires but nothing mutes → confirmation failed; the ad profile did
  not survive `confirm_windows`. Lower `confirm_windows` or relax the profile
  thresholds.
* Ad ends late, running into dialogue → lower `detection.ad_end_windows`, or
  `min_ad_seconds`.
* Ad ends early during a soft passage *inside* one spot → lower
  `ad_stay_loudness_delta_db`. Check `loudness_delta_db` against
  `loudness_threshold_db` in the metrics: if the delta is positive but under the
  threshold while the ad is running, the stay bar is what ended it.
* Ad ends early, mid-break → raise `ad_end_windows` (a silent seam between two
  spots, or a quiet moment inside one, was read as content). On the recording
  measured, 2 was far too low and 5–6 was the zero-false-mute range — but that
  range was narrow, with 4 leaking a spurious mute and 8 overhanging, so re-run
  the sweep rather than assuming 6 transfers.

## Known limitations

* **Mute is a toggle.** ECP has no discrete mute/unmute and no way to query mute
  state, so the client tracks what it believes. If someone hits Mute on the
  remote while the service is running, the two can desync — restart the service
  while the TV is unmuted to re-sync (or set `roku.assume_muted_at_start: true`
  if it is muted at start).
* **Netflix gating is best-effort.** `query/active-app` tells us Netflix is on
  screen, not that something is playing. If the query fails, the previous armed
  state is kept rather than guessing.
* **Heuristics, not understanding.** Phase 1 knows about loudness, dynamics, and
  seams. Trailers before content, and ads mastered gently, will be missed.
* **Roughly half of a caught break still plays.** On the one recording scored,
  the tuned config mutes 51% of ad audio: it mutes at the start of a break, and
  after an inter-spot seam it often does not re-acquire. Raising coverage past
  this point with the current thresholds means overhanging into returning
  dialogue, which costs more than it buys. Holding a mute through a seam and
  re-acquiring after one need to be separately controllable to do better —
  that is a detector change, not a config change.
* **Tuned on a single recording.** The shipped thresholds come from three ad
  breaks in one movie on one service. They are a starting point; see
  [the tuning guide](#what-one-labelled-recording-actually-showed).
* **The baseline needs warm-up.** After a start, a capture restart, or Netflix
  becoming active, `baseline_min_windows` (15 s by default) pass before anything
  can fire.

## Phase 2 hooks

`detector.Detector` is a Protocol with three methods (`update`, `reset`,
`reject`). A classifier that returns a `Decision` — with both the `event` edge
and the `ad_profile` level set, per [The model](#the-model) — drops into
`__main__.py` in place of `HeuristicDetector` with no changes to capture,
features, controller, or the service unit. The feature log written by Phase 1 is
the training data.

Two constraints on any replacement:

* **No clock calls.** `update` is handed a timestamp; it must never read
  `time.monotonic()` itself. That is what lets `replay_wav.py` reproduce live
  behaviour exactly from a WAV file.
* **`reject()` must clear ad state without clearing what was learned.** The
  controller calls it when it overrules an `AD_STARTED` (confirmation failed, or
  the mute failsafe tripped). A detector that ignores it will sit in its ad state
  and miss the next real transition.

One place the seam is tight: `update` receives a `Features` — fourteen scalars —
not raw audio. That suits a tree ensemble or a small MLP over tabular features.
A model wanting spectrograms needs the samples to cross the boundary, most
cleanly by adding mel bands to `Features` in `features.py`, which keeps the
protocol and the feature log intact.
