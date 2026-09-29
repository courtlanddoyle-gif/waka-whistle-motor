# Waka Waka whistle → LEGO motor

Whistle the opening of *Waka Waka* at your laptop and a LEGO Education Single
Motor spins while the song plays.

Built for Tufts ME193, *AI in Robotics*.

## How it recognizes the tune

Matching audio to audio doesn't work for whistling — you never whistle it in
quite the same key, at quite the same speed, or quite in tune. So the script
throws away everything except the shape of the melody:

1. **Pitch tracking.** Every ~23 ms, an FFT finds the strongest frequency and
   converts it to a note number. A frame only counts as whistling if that peak
   really stands out — measured against a noise floor the script calibrates at
   startup and keeps adapting, plus checks on how far the peak rises above the
   rest of the band and how much of the energy sits in it. Judging *peakiness*
   rather than loudness is what stops a door slam or someone talking from
   deafening it for seconds.
2. **Cleanup.** One-frame blips are dropped, short dropouts inside a held note
   are bridged, and single-frame glitches are median-filtered. Anything outside
   the octave where the whistling actually lives is discarded — otherwise a
   steady hum in the room becomes a "note" of the reference that you then have
   to reproduce to get a match.
3. **Matching.** Dynamic time warping compares the last few seconds against the
   reference, allowing for a different key and a different tempo. Every note of
   the reference is weighted to count about equally and must be matched on its
   own, so a missing, swapped, or wrong note fails even when the average looks
   fine. Matching runs on every new whistled frame, so the tune is caught as its
   last note starts rather than after you stop.

The reference is a recording of **you** whistling, not the Shakira track — a
whistle detector looks for one clean tone at a time, and against the real
recording (drums, bass, vocals, horns together) only about 20% of frames read as
a pure tone. Your own recording also matches your own whistling far better.

## Setup

Needs Python 3.10+ and a LEGO Education Single Motor with its Connection Card.

```bash
python -m venv .venv
# Windows:  .venv\Scripts\Activate.ps1
# macOS:    source .venv/bin/activate
pip install numpy sounddevice soundfile legoeducation
```

Two things are **not** in this repo and you supply yourself:

- **`lelib.py`** — the class's LEGO helper library, from the
  [ME193-Robotics](https://github.com/chrisbuerginrogers/ME193-Robotics) repo
  (`Public stuff/useful libraries/`). Drop it in this folder.
- **The MP3** — put any audio file in this folder and it's picked up
  automatically; name it `waka_waka.mp3` to be explicit.

Then set `CARD_COLOR_NAME` and `CARD_SERIAL` near the top of the script to match
the Connection Card on your motor.

## Use

```bash
python waka_motor_control.py --record   # record your reference whistle, once
python waka_motor_control.py --check    # how close was that? no motor, just the number
python waka_motor_control.py            # listen for real
```

While it's listening, these keys work without pressing Enter:

| Key | Does what |
| --- | --- |
| `p` | Pause — stops listening and stops whatever is playing, but stays connected to the motor. Press again to resume. |
| `q` | Stop cleanly (same as Ctrl+C). |

Resuming forgets whatever was heard before the pause, so half a tune from either
side can't combine into a false match. The microphone keeps being read while
paused, so the noise floor stays current and resuming is instant.

`--record` reports whether the recording is clipping, how many notes it found,
and whether it leaves enough margin to be matched again. `--check` records one
attempt and prints its match cost against the reference, with advice on whether
to adjust the threshold or re-record.

Other flags: `--verbose` prints the live match cost while you whistle,
`--test FILE` runs the detector over an audio file, and `--no-motor` skips
Bluetooth so you can work on the audio side alone.

## Also here: `whistle_motor_control.py`

A simpler, earlier script that drives the same motor from whistle *pitch* rather
than melody — no reference recording, no matching:

| You whistle | Motor does |
| --- | --- |
| high | spins clockwise |
| low | spins counterclockwise |
| in the middle | flips direction every 0.2 s |
| nothing | stops |

It takes the loudest frequency in each audio block straight from an FFT and
buckets it, which is about as simple as pitch control gets — worth reading first
if the DTW matching in the main script is a lot to take in at once.

Its `LOW_PITCH_MAX_HZ` (800) and `HIGH_PITCH_MIN_HZ` (1600) are **placeholders**,
not tuned to anyone's voice. Run `waka_motor_control.py --check` and read the
frequency range it reports for your whistling, then set them from that.

```bash
python whistle_motor_control.py
```

## Tuning

Everything worth adjusting is a named constant at the top of the script.

| Constant | Does what |
| --- | --- |
| `MATCH_THRESHOLD` | Match cost at or below which it fires. Higher = easier to set off, and easier to set off by accident. |
| `MIN_FREQ` / `MAX_FREQ` | The whistle band. Lower `MIN_FREQ` if you whistle low and notes go missing entirely. |
| `MAX_TUNE_SPAN` | How wide a pitch window the tune is assumed to fit in (6 = one octave). |
| `MOTOR_SPEED` | Motor speed, 0–100. |
| `SPIN_FOR_WHOLE_SONG` | Spin for the song's full length, or `SPIN_SECONDS` instead. |

Use `--check` to tune `MATCH_THRESHOLD` with real numbers rather than guessing.
After raising it, deliberately whistle the *wrong* tune and confirm it still
misses — that's the check that matters.
