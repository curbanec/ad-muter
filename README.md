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

git clone <this repo> ~/netflix-ad-muter
cd ~/netflix-ad-muter
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt        # add -r requirements-dev.txt for pytest
```

If you would rather not use a venv, `pip install --break-system-packages -r
requirements.txt` works, but then the systemd unit must call `/usr/bin/python3`
instead of `.venv/bin/python`.

Dependencies are `sounddevice`, `numpy`, `requests`, `PyYAML` — nothing else.
Python ≥ 3.11.

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
INFO  admuter.controller  AD_SUSPECTED (1/2) conf=0.92 — gap=0.50s then loudness +6.1dB / crest -4.3dB vs baseline
INFO  admuter.controller  MUTE (2/2 windows) — gap=0.50s then loudness +6.1dB / crest -4.3dB vs baseline
INFO  admuter.controller  UNMUTE — 2 non-ad windows after 61s
```

Per-window feature lines are DEBUG. Steady-state `NO_CHANGE` decisions are
deduplicated to DEBUG with an INFO heartbeat every `decision_heartbeat_windows`
windows, so the journal does not grow by one line per second forever; set
`logging.verbose_decisions: true` if you want literally every decision at INFO.

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

The unit assumes the repo at `/home/chris/netflix-ad-muter` with its venv at
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
.venv/bin/python scripts/replay_wav.py sample.wav --set detection.ad_crest_delta_db=3.0
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

**Collecting training data in production**: set `logging.feature_log_enabled:
true` and every window is appended to `logs/features.jsonl` (or `.csv`) with its
features, the detector's metrics, the decision, and the state. Annotate the ad
spans later and you have a labelled set for Phase 2.

## Tuning guide

Two failure modes, opposite fixes. Change **one knob at a time** and re-run
`replay_wav.py` over your saved samples — that is much faster than waiting for
the next real ad break.

### Failure mode 1: it mutes real content (the serious one)

A loud, compressed scene right after a quiet one can look like an ad. In order of
what to reach for:

| Knob | Direction | Effect |
| --- | --- | --- |
| `controller.confirm_windows` | ↑ 2 → 3 | Requires the ad profile to hold for another second. Cheapest, safest fix; costs ~1 s of extra latency. |
| `detection.ad_crest_delta_db` | ↑ 2 → 3–4 | Demands genuinely squashed audio. Crest factor is the most discriminating single feature — real ads sit far below show audio. |
| `detection.ad_loudness_delta_db` | ↑ 2 → 3–4 | Demands a bigger loudness jump over the baseline. |
| `detection.require_transition_cue` | keep `true` | With this off, loudness alone can fire. Only turn it off if you are missing nearly every ad. |
| `detection.max_gap_seconds` | ↓ 1.5 → 0.8 | Long pauses in a quiet scene stop counting as seams. |
| `detection.baseline_alpha` | ↓ 0.05 → 0.02 | Slower baseline, less influenced by a recent loud stretch. |

If a false mute does slip through, `detection.max_ad_seconds` and
`controller.max_mute_seconds` bound the damage — with the defaults the TV can
never stay muted for more than ~130 s.

### Failure mode 2: it misses ads

| Knob | Direction | Effect |
| --- | --- | --- |
| `detection.ad_crest_delta_db` | ↓ 2 → 1.0–1.5 | Accepts ads that are only somewhat more compressed than the show. |
| `detection.ad_loudness_delta_db` | ↓ 2 → 1.0–1.5 | Netflix normalizes loudness fairly well, so this delta can be genuinely small. |
| `controller.confirm_windows` | ↓ 2 → 1 | Mutes on the first ad-looking window. Faster, noticeably riskier. |
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
* Ad ends late → lower `detection.ad_end_windows` to 1, or `min_ad_seconds`.
* Ad ends early, mid-break → raise `ad_end_windows` to 3 (a quiet moment inside
  an ad was read as content).

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
* **The baseline needs warm-up.** After a start, a capture restart, or Netflix
  becoming active, `baseline_min_windows` (15 s by default) pass before anything
  can fire.

## Phase 2 hooks

`detector.Detector` is a Protocol with three methods (`update`, `reset`,
`reject`). A classifier that returns a `Decision` — with both the `event` edge
and the `ad_profile` level set — drops into `__main__.py` in place of
`HeuristicDetector` with no changes to capture, features, controller, or the
service unit. The feature log written by Phase 1 is the training data.
