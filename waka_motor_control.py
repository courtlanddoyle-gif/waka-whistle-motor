"""
Recognize the first few notes of "Waka Waka" whistled into the microphone.
When it hears them: spin the single motor and play an MP3.

Built on the same ideas as the Rock-a-bye Baby recognizer:

  o The reference is turned into a *pitch contour* -- one pitch (in semitones)
    per short frame of audio, with gaps where nothing is whistled.
  o The microphone is tracked exactly the same way, and the most recent stretch
    of whistling is compared to the reference with dynamic time warping (DTW),
    so it matches in any key and at somewhat different tempos.
  o Pitch tracking keeps a per-frequency noise floor (calibrated at startup,
    then adapted continuously), so fans, hum and laptop whine are subtracted
    out instead of drowning the whistle.
  o A frame counts as "whistle" based on how far its peak stands out (SNR above
    the noise floor, prominence over the rest of the band, and purity), not on
    raw volume -- so a clap or someone talking doesn't deafen it for seconds.
  o Frames overlap (a new frame every ~23 ms), doubling time resolution.
  o The contour is cleaned before matching: one-frame blips dropped, short
    dropouts inside a note bridged, single-frame pitch glitches median-filtered.
  o Every note of the reference counts about equally and each must be matched on
    its own, so a missing, swapped or wrong note fails even when the average
    looks fine. That makes false matches rare.
  o Matching runs on every new whistled frame, so the tune is caught as its last
    note starts rather than after you stop whistling.

Differences from that script, for this setup:
  o Audio in/out goes through sounddevice (the class standard) instead of
    pyaudio, and MP3/WAV decoding through soundfile instead of macOS afconvert.
  o The audio thread only tracks pitch; all Bluetooth motor commands are issued
    from the main loop, so a slow BLE write can never stall the microphone.
  o The reference is a recording of YOU whistling the hook. Don't point this at
    the real Shakira track -- that recording is drums, bass and vocals all at
    once, and a whistle detector (which looks for one clean tone) can't track
    it. Record your own, once, and it matches your whistle far better anyway.

First time -- record your reference:
    python waka_motor_control.py --record

Then drop your MP3 in this folder and run:
    python waka_motor_control.py

While it's listening, press 'p' to pause (it goes quiet and still, and stops
whatever it's playing, but stays connected to the motor), 'p' again to resume,
and 'q' to stop.

If it won't trigger, don't guess -- measure. This records one attempt and tells
you the match cost and what to do about it:
    python waka_motor_control.py --check

Other modes:
    python waka_motor_control.py --verbose        show the match cost as you whistle
    python waka_motor_control.py --test FILE      run the detector over an audio file
    python waka_motor_control.py --no-motor       audio only, don't connect over Bluetooth

Requires: pip install numpy sounddevice soundfile legoeducation
"""

import argparse
import os
import queue
import sys
import time
from collections import deque
from pathlib import Path

import numpy as np
import sounddevice as sd
import soundfile as sf

HERE = Path(__file__).parent

# Your own whistled reference, made by --record.
REFERENCE = HERE / "waka_reference.wav"
REFERENCE_SECONDS = 5.0  # how long --record listens for
ATTEMPT = HERE / "waka_attempt.wav"  # where --check saves what it heard

# The MP3 to play on a match. If this exact name isn't here, the script picks
# whichever audio file in the folder isn't the reference recording.
SONG_FILE = HERE / "waka_waka.mp3"

# --- LEGO hardware --------------------------------------------------------
CARD_COLOR_NAME = "PURPLE"
CARD_SERIAL = 6235
MOTOR_SPEED = 60            # percent, 0-100
SPIN_FOR_WHOLE_SONG = True  # True = spin until the MP3 finishes
SPIN_SECONDS = 5.0          # ...otherwise, spin this long (also the fallback
                            # if no song file is found)

# --- Keys you can press while it's listening ------------------------------
PAUSE_KEY = "p"  # stop/resume listening, without disconnecting the motor
QUIT_KEY = "q"   # stop cleanly, same as Ctrl+C

# --- Audio framing --------------------------------------------------------
RATE = 44100
HOP = 1024          # a new frame every ~23 ms
WINDOW = 2048       # each frame looks at the last ~46 ms
FFT_SIZE = WINDOW * 4  # zero-padding, for finer peak location

# Whistling is almost a pure sine wave, typically between ~500 Hz and ~4 kHz.
# Notes outside this band are never tracked at all, so if you whistle the tune
# low and --verbose shows a note going missing (the cost never drops), lower
# MIN_FREQ rather than raising MATCH_THRESHOLD.
MIN_FREQ = 500
MAX_FREQ = 4500

# A frame counts as "whistle" when its strongest peak...
MIN_RMS = 0.0005          # (frame isn't digital silence, 0-1 full scale)
MIN_SNR_DB = 12.0         # ...is this far above the room's noise at that frequency,
MIN_PROMINENCE_DB = 20.0  # ...this far above the band's median level,
PURITY_THRESHOLD = 0.5    # ...and holds this share of the (noise-subtracted) band energy
PURITY_HALF_WIDTH = 60.0  # Hz either side of the peak counted as "the peak"
NOISE_OVERSUBTRACT = 2.0  # noise floor is multiplied by this before subtracting
NOISE_RISE = 0.03         # per-frame adaptation of the noise floor when it goes up
NOISE_FALL = 0.3          # ...and when it goes down (fast, so a bad calibration heals)
CALIBRATION_SECONDS = 1.0  # room noise recorded at startup (stay quiet!)

# --- Contour cleanup (in frames of HOP samples) ---------------------------
MIN_RUN = 2   # whistled stretches shorter than this are dropped as blips
MAX_FILL = 5  # dropouts up to this long (~115 ms) between whistled frames are bridged

# Anything that isn't the tune -- a fan, a hum, a chair, your own breath before
# you start -- can still look like a clean tone and become a "note". In the
# reference that's poison: every note of the reference has to be reproduced for
# a match, so a note of room noise makes the tune impossible to whistle again.
# So keep only the pitch range the whistling actually lives in, and drop notes
# too short to be notes.
MAX_TUNE_SPAN = 6.0   # half-width, in semitones, of the pitch window the tune
                      # is assumed to fit in -- 6 means one octave. Raise it
                      # only if your tune really does span more than an octave.
MIN_NOTE_FRAMES = 3   # a note shorter than ~70 ms is a blip, not a note

# --- Matching -------------------------------------------------------------
# Silent frames are kept in both contours, so rhythm counts too.
TRANSPOSE_INVARIANT = True  # True = whistling it in any key counts
TEMPO_STRETCHES = np.linspace(0.7, 1.45, 8)  # window lengths tried, relative to the reference
MAX_STEP_COST = 3.0    # semitones; caps the penalty for a wild frame
GAP_COST = 1.0         # penalty for whistle vs. silence in the same step
WARP_COST = 0.3        # extra penalty for stretching one side (non-diagonal step)
WARP_BAND = 0.2        # DTW path must stay within this fraction of the diagonal
NOTE_SPLIT = 1.0       # semitones; a bigger jump starts a new note
MAX_NOTE_WEIGHT = 3.0  # limit on how much more a short note counts than a long one
REST_WEIGHT = 0.25     # weight of a silent reference frame (a note frame averages 1)
MAX_NOTE_COST = 1.25   # semitones; a reference note matched worse than this on average...
NOTE_PENALTY = 1.0     # ...adds this much per semitone beyond it to the cost
MATCH_THRESHOLD = 0.75  # avg DTW cost (semitones) at or below = match. Higher
                        # is easier to set off -- and easier to set off by
                        # accident; lower demands a closer whistle. After
                        # changing it, use --check both ways: that your real
                        # whistle still fires, and a WRONG tune still misses.
MIN_VOICED_RATIO = 0.6  # window must have at least this share of the reference's
                        # whistled frames
COOLDOWN_S = 2.0        # ignore new matches this long after one fires


# ---------------------------------------------------------------- audio I/O

def load_audio(path):
    """Decode any file soundfile understands to mono float32 at RATE."""
    data, rate = sf.read(str(path), dtype="float32", always_2d=True)
    mono = data.mean(axis=1)
    if rate != RATE:  # linear resample is plenty for tracking a whistle
        n = int(round(len(mono) * RATE / rate))
        mono = np.interp(np.linspace(0, len(mono) - 1, n),
                         np.arange(len(mono)), mono).astype(np.float32)
    return mono


def frames_of(audio):
    """Overlapping WINDOW-long frames, one every HOP samples."""
    for i in range(0, len(audio) - WINDOW + 1, HOP):
        yield audio[i:i + WINDOW]


def find_song():
    """The MP3 to play: SONG_FILE if it's there, else any other audio file in
    the folder that isn't the reference recording."""
    if SONG_FILE.exists():
        return SONG_FILE
    ours = {REFERENCE.name.lower(), ATTEMPT.name.lower()}  # our own recordings
    for path in sorted(HERE.iterdir()):
        if path.suffix.lower() in (".mp3", ".wav", ".m4a", ".ogg", ".flac"):
            if path.name.lower() not in ours:
                return path
    return None


# ------------------------------------------------------------ pitch tracking

_window = np.hanning(WINDOW)
_freqs = np.fft.rfftfreq(FFT_SIZE, 1 / RATE)
_band = (_freqs >= MIN_FREQ) & (_freqs <= MAX_FREQ)
_band_freqs = _freqs[_band]
_bin_hz = _freqs[1] - _freqs[0]
_half = int(round(PURITY_HALF_WIDTH / _bin_hz))


def band_power(samples):
    return np.abs(np.fft.rfft(samples * _window, n=FFT_SIZE))[_band] ** 2


class PitchTracker:
    """Turns audio frames into whistle pitches (MIDI number, or None)."""

    def __init__(self):
        self.noise = None  # per-bin noise power in the whistle band

    def calibrate(self, frames):
        self.noise = np.mean([band_power(f) for f in frames], axis=0)

    def process(self, samples):
        power = band_power(samples)
        pitch = self._detect(samples, power)
        if pitch is None:
            self._update_noise(power)
        return pitch

    def _detect(self, samples, power):
        if np.sqrt(np.mean(samples ** 2)) < MIN_RMS:
            return None
        noise = self.noise if self.noise is not None else np.zeros_like(power)
        clean = np.maximum(power - NOISE_OVERSUBTRACT * noise, 0.0)
        peak = int(np.argmax(clean))
        peak_power = power[peak]
        if peak_power < 10 ** (MIN_SNR_DB / 10) * noise[peak]:
            return None
        if peak_power < 10 ** (MIN_PROMINENCE_DB / 10) * np.median(power):
            return None
        total = clean.sum()
        if total <= 0 or clean[max(peak - _half, 0):peak + _half + 1].sum() < PURITY_THRESHOLD * total:
            return None
        # Parabolic interpolation around the peak, for sub-bin accuracy.
        offset = 0.0
        if 0 < peak < len(power) - 1:
            a, b, c = 0.5 * np.log(power[peak - 1:peak + 2] + 1e-20)
            denom = a - 2 * b + c
            if denom < 0:
                offset = 0.5 * (a - c) / denom
        freq = _band_freqs[peak] + offset * _bin_hz
        return 69 + 12 * np.log2(freq / 440.0)

    def _update_noise(self, power):
        if self.noise is None:
            self.noise = power.copy()
            return
        # Follow drops quickly; rise slowly, and never by more than 4x per
        # frame, so the tail of a whistle can't inflate the floor.
        target = np.minimum(power, 4 * self.noise)
        rate = np.where(power < self.noise, NOISE_FALL, NOISE_RISE)
        self.noise += rate * (target - self.noise)


def pitch_track(audio):
    """Per-frame pitch (MIDI or None) of a whole recording."""
    tracker = PitchTracker()
    return [tracker.process(frame) for frame in frames_of(audio)]


def build_template(path):
    """Reference contour from first to last whistled frame (NaN = silence)."""
    track = drop_stray_notes(clean_contour(to_array(pitch_track(load_audio(path)))))
    voiced = np.flatnonzero(~np.isnan(track))
    if len(voiced) == 0:
        raise SystemExit(
            f"No whistle found in {path}.\n"
            "Re-record it with:  python waka_motor_control.py --record\n"
            "and whistle steadily and fairly close to the microphone."
        )
    return track[voiced[0]:voiced[-1] + 1]


# ------------------------------------------------------------------ matching

def to_array(pitches):
    return np.array([np.nan if p is None else p for p in pitches], dtype=float)


def _runs(mask):
    """(start, end) index pairs of the True runs in a boolean array."""
    edges = np.diff(np.concatenate(([0], mask.astype(np.int8), [0])))
    return zip(np.flatnonzero(edges == 1), np.flatnonzero(edges == -1))


def clean_contour(seq):
    """Drop blips, bridge short dropouts, median-filter single-frame glitches."""
    seq = seq.copy()
    for start, end in _runs(~np.isnan(seq)):
        if end - start < MIN_RUN:
            seq[start:end] = np.nan
    for start, end in _runs(np.isnan(seq)):
        if 0 < start and end < len(seq) and end - start <= MAX_FILL:
            mid = (start + end) // 2  # each side of the gap holds its neighbour's pitch
            seq[start:mid] = seq[start - 1]
            seq[mid:end] = seq[end]
    # Median of 3; at the edge of a whistled stretch the missing neighbour is
    # replaced by the one on the other side, so onset/offset glitches go too.
    padded = np.concatenate(([np.nan], seq, [np.nan]))
    prev, nxt = padded[:-2], padded[2:]
    prev, nxt = np.where(np.isnan(prev), nxt, prev), np.where(np.isnan(nxt), prev, nxt)
    both = ~np.isnan(seq) & ~np.isnan(prev)
    seq[both] = np.median(np.stack((prev, seq, nxt))[:, both], axis=0)
    return seq


def _tune_band(seq, span):
    """The pitch window of width 2*span holding the most whistled frames --
    i.e. where the tune actually lives, as opposed to stray noise elsewhere in
    the band."""
    voiced = np.sort(seq[~np.isnan(seq)])
    if len(voiced) == 0:
        return None
    reach = np.searchsorted(voiced, voiced + 2 * span, side="right")
    start = int(np.argmax(reach - np.arange(len(voiced))))
    return voiced[start], voiced[start] + 2 * span


def drop_stray_notes(seq):
    """Blank out everything that isn't part of the tune: frames far from the
    tune's own pitch range, and notes too short to be real notes.

    Without this, steady room noise becomes a note of the reference that every
    later whistle is required to reproduce -- and, live, a single stray frame
    drags normalize()'s range midpoint away and throws the whole window off
    key."""
    seq = seq.copy()
    band = _tune_band(seq, MAX_TUNE_SPAN)
    if band is None:
        return seq
    lo, hi = band
    seq[(seq < lo) | (seq > hi)] = np.nan
    ids = note_ids(seq)
    for nid in np.unique(ids[ids >= 0]):
        if np.count_nonzero(ids == nid) < MIN_NOTE_FRAMES:
            seq[ids == nid] = np.nan
    return seq


def normalize(seq):
    """Shift a contour so the middle of its pitch range is 0 (if key doesn't
    matter). The midpoint of lowest and highest note doesn't depend on how long
    each note is held, unlike the median."""
    if not TRANSPOSE_INVARIANT:
        return seq
    voiced = seq[~np.isnan(seq)]
    return seq - 0.5 * (voiced.min() + voiced.max())


def note_ids(seq):
    """Number the notes of a contour (-1 for rests). A new note starts wherever
    the pitch jumps by more than NOTE_SPLIT semitones."""
    silent = np.isnan(seq)
    change = np.abs(np.diff(seq)) > NOTE_SPLIT
    change |= silent[1:] != silent[:-1]
    ids = np.concatenate(([0], np.cumsum(change)))
    ids[silent] = -1
    return ids


def note_weights(notes):
    """Per-frame weight that makes every note count about the same, so a short
    note -- like the tune's last one -- can't just be skipped. Rests count
    little, since whether you pause between notes is mostly style."""
    voiced = notes >= 0
    lengths = np.bincount(notes[voiced])[notes[voiced]]
    weights = np.full(len(notes), REST_WEIGHT)
    weights[voiced] = np.clip(lengths.mean() / lengths, 1 / MAX_NOTE_WEIGHT, MAX_NOTE_WEIGHT)
    return weights / weights.mean()


def local_costs(a, b):
    """Frame-vs-frame distance between two contours (NaN = silence)."""
    local = np.minimum(np.abs(a[:, None] - b[None, :]), MAX_STEP_COST)
    silent_a, silent_b = np.isnan(a)[:, None], np.isnan(b)[None, :]
    local[silent_a | silent_b] = GAP_COST
    local[silent_a & silent_b] = 0.0
    return local


def dtw(a, b, notes, weights):
    """Score how well contour b matches template a, plus the median pitch
    difference (b - a) along the best alignment.

    The score is the average per-frame DTW distance (a's frames weighted by
    `weights`), plus a penalty for every note of a (numbered by `notes`) that is
    off by more than MAX_NOTE_COST on average -- so a missing, extra or swapped
    note fails even when the rest of the tune fits.

    Uses the symmetric step pattern (1,1), (1,2), (2,1), which limits local
    tempo changes to 2x and -- because each row only depends on the two rows
    above -- lets every row be computed at once with numpy."""
    n, m = len(a), len(b)
    raw = local_costs(a, b)
    local = raw * weights[:, None]
    i_idx, j_idx = np.arange(n)[:, None], np.arange(m)[None, :]
    local[np.abs(i_idx / n - j_idx / m) > WARP_BAND] = np.inf
    left = np.concatenate((np.full((n, 1), np.inf), local[:, :-1]), axis=1)  # local[i, j-1]

    acc = np.full((n + 2, m + 2), np.inf)  # acc[i+2, j+2] = best cost ending at (i, j)
    acc[1, 1] = 0.0
    choice = np.zeros((n, m), dtype=np.int8)
    for i in range(n):
        r = i + 2
        up = local[i - 1] if i > 0 else np.full(m, np.inf)
        options = np.stack((acc[r - 1, 1:-1] + 2 * local[i],                      # (i-1, j-1)
                            acc[r - 1, :-2] + 2 * left[i] + local[i] + WARP_COST,  # (i-1, j-2)
                            acc[r - 2, 1:-1] + 2 * up + local[i] + WARP_COST))     # (i-2, j-1)
        choice[i] = np.argmin(options, axis=0)
        acc[r, 2:] = options[choice[i], np.arange(m)]

    total = acc[n + 1, m + 1]
    if not np.isfinite(total):
        return np.inf, 0.0

    # Walk the best path back to collect aligned frame pairs.
    pairs, i, j = [], n - 1, m - 1
    while i >= 0 and j >= 0:
        pairs.append((i, j))
        step = choice[i, j]
        if step == 0:
            i, j = i - 1, j - 1
        elif step == 1:
            pairs.append((i, j - 1))
            i, j = i - 1, j - 2
        else:
            pairs.append((i - 1, j))
            i, j = i - 2, j - 1

    ia, jb = np.array(pairs).T
    diffs = b[jb] - a[ia]
    diffs = diffs[~np.isnan(diffs)]
    shift = float(np.median(diffs)) if len(diffs) else 0.0

    # For the per-note check, a note met by silence counts as fully missed.
    on_note = notes[ia] >= 0
    frame_cost = np.where(np.isnan(b[jb]), MAX_STEP_COST, raw[ia, jb])[on_note]
    per_note = (np.bincount(notes[ia][on_note], frame_cost)
                / np.maximum(np.bincount(notes[ia][on_note]), 1))
    miss = np.maximum(per_note - MAX_NOTE_COST, 0.0).sum()
    return total / (n + m) + NOTE_PENALTY * miss, shift


def match_cost(template, notes, weights, window):
    """DTW score of a (normalized) live window against the template, with the
    key refined once from the alignment."""
    cost, shift = dtw(template, window, notes, weights)
    if TRANSPOSE_INVARIANT and np.isfinite(cost) and abs(shift) > 0.2:
        cost = min(cost, dtw(template, window - shift, notes, weights)[0])
    return cost


class TuneMatcher:
    """Feed one frame's pitch at a time; reports when the tune has been heard."""

    def __init__(self, template):
        self.template = normalize(template)
        self.notes = note_ids(template)
        self.weights = note_weights(self.notes)
        self.template_voiced = np.count_nonzero(~np.isnan(template))
        self.history = deque(maxlen=int(len(template) * max(TEMPO_STRETCHES)) + MAX_FILL + 4)
        self.cooldown_until = -np.inf
        self.last_best = None

    def update(self, pitch, now):
        """Return the match cost if the tune was just recognized, else None."""
        self.history.append(pitch)
        # Check on every whistled frame, so the tune is caught during its last note.
        if pitch is None or now < self.cooldown_until:
            return None
        best = self.best_cost()
        self.last_best = best
        if best is not None and best <= MATCH_THRESHOLD:
            self.cooldown_until = now + COOLDOWN_S
            self.history.clear()
            return best
        return None

    def mute_until(self, when):
        """Ignore everything heard until `when` -- used while the song is
        playing, so the speakers can't set it off again."""
        self.cooldown_until = max(self.cooldown_until, when)
        self.history.clear()

    def best_cost(self):
        """Lowest DTW cost over the tempo stretches, or None if not enough whistle."""
        # Same stray-note cleanup as the reference got, so the two contours are
        # treated alike and a bit of room noise can't shift the window off key.
        frames = drop_stray_notes(clean_contour(to_array(self.history)))
        voiced = np.flatnonzero(~np.isnan(frames))
        if len(voiced) == 0:
            return None
        frames = frames[:voiced[-1] + 1]  # cleanup may have dropped a trailing blip
        best, tried = None, set()
        for stretch in TEMPO_STRETCHES:
            window = frames[-int(round(len(self.template) * stretch)):]
            is_voiced = ~np.isnan(window)
            if np.count_nonzero(is_voiced) < MIN_VOICED_RATIO * self.template_voiced * stretch:
                continue
            # Start the window at its first whistled frame, like the template.
            start = np.flatnonzero(is_voiced)[0]
            if len(window) - start in tried:
                continue
            tried.add(len(window) - start)
            cost = match_cost(self.template, self.notes, self.weights,
                              normalize(window[start:]))
            best = cost if best is None else min(best, cost)
        return best


# -------------------------------------------------------------- the reaction

class Reaction:
    """What happens on a match: the motor spins and the song plays. Both are
    started here and left running; `tick()` stops the motor when its time is
    up, so the main loop never blocks waiting on either one."""

    def __init__(self, motor, direction, song_path):
        self.motor = motor
        self.direction = direction
        self.song = None
        self.song_rate = RATE
        self.song_seconds = SPIN_SECONDS
        self.spin_until = None
        if song_path is not None:
            self.song, self.song_rate = sf.read(str(song_path), dtype="float32")
            self.song_seconds = len(self.song) / self.song_rate

    def fire(self, now):
        """Start the motor and the song. Returns when the reaction ends."""
        seconds = self.song_seconds if (SPIN_FOR_WHOLE_SONG and self.song is not None) \
            else SPIN_SECONDS
        if self.song is not None:
            sd.play(self.song, self.song_rate)
        if self.motor is not None:
            self.motor.motor_run(direction=self.direction, speed=MOTOR_SPEED)
        self.spin_until = now + seconds
        return self.spin_until

    def tick(self, now):
        if self.spin_until is not None and now >= self.spin_until:
            self.spin_until = None
            if self.motor is not None:
                self.motor.motor_stop()

    def stop(self):
        sd.stop()
        if self.motor is not None and self.spin_until is not None:
            self.motor.motor_stop()
        self.spin_until = None


# --------------------------------------------------------------------- modes

class Keyboard:
    """Reads single keypresses without waiting for Enter and without blocking
    the listening loop -- get() returns None when nothing has been pressed.

    If the terminal can't be put into that mode (input is piped, or there's no
    terminal at all), `enabled` stays False and get() simply never reports a
    key, so the script still runs -- just without the hotkeys."""

    def __init__(self):
        self.enabled = False
        self._restore = None
        self._read = lambda: None

    def __enter__(self):
        if os.name == "nt":
            try:
                import msvcrt
                self._read = lambda: msvcrt.getwch() if msvcrt.kbhit() else None
                self.enabled = True
            except Exception:
                pass
        else:
            try:
                import select
                import termios
                import tty
                fd = sys.stdin.fileno()
                saved = termios.tcgetattr(fd)
                tty.setcbreak(fd)  # deliver keys as typed, but keep Ctrl+C working
                self._restore = lambda: termios.tcsetattr(fd, termios.TCSADRAIN, saved)
                self._read = lambda: (sys.stdin.read(1)
                                      if select.select([sys.stdin], [], [], 0)[0]
                                      else None)
                self.enabled = True
            except Exception:
                pass
        return self

    def get(self):
        try:
            key = self._read()
        except Exception:
            return None
        return key.lower() if key else None

    def __exit__(self, *exc):
        if self._restore:
            self._restore()


def announce(cost, when):
    print(f"\n*** Waka Waka recognized at {when} (cost {cost:.2f}) ***")


def record_to(path, seconds, prompt):
    """Count down, record, and save."""
    print(f"{prompt} Recording {seconds:g} seconds.")
    for count in (3, 2, 1):
        print(f"  {count}...")
        time.sleep(1.0)
    print("  GO -- whistle now!")
    audio = sd.rec(int(seconds * RATE), samplerate=RATE, channels=1, dtype="float32")
    sd.wait()
    sf.write(str(path), audio, RATE)
    print(f"Saved {path.name}.")
    return audio[:, 0]


def describe(audio, template):
    """Report what the detector actually made of a recording, and complain
    about the things that quietly ruin a match."""
    peak = float(np.abs(audio).max())
    print(f"\n  level: peak {peak:.2f}", end="")
    if peak >= 0.99:
        print("  <-- CLIPPING. Turn the mic gain down, or back off from it;"
              "\n         a clipped whistle isn't a clean tone any more.")
    elif peak < 0.05:
        print("  <-- very quiet. Move closer to the mic.")
    else:
        print("  (good)")

    seconds = len(template) * HOP / RATE
    ids = note_ids(template)
    notes = len(np.unique(ids[ids >= 0]))
    voiced = template[~np.isnan(template)]
    if len(voiced) == 0:
        print("  tune:  no whistle detected at all in this recording.")
        return
    span = float(voiced.max() - voiced.min())
    print(f"  tune:  {notes} notes over {seconds:.2f} s, "
          f"spanning {span:.1f} semitones "
          f"({440 * 2 ** ((voiced.min() - 69) / 12):.0f}-"
          f"{440 * 2 ** ((voiced.max() - 69) / 12):.0f} Hz)")
    if seconds < 1.0:
        print("         <-- very short; short tunes trigger by accident.")
    if notes < 4:
        print("         <-- few distinct notes; whistle more of the hook.")
    if span < 2.0:
        print("         <-- nearly all one pitch, so almost any held whistle\n"
              "             will match it. Whistle the part where the tune moves.")


def best_cost_over(audio_path, template):
    """Lowest match cost the detector reaches anywhere in a recording."""
    matcher = TuneMatcher(template)
    best = np.inf
    for k, pitch in enumerate(pitch_track(load_audio(audio_path))):
        matcher.update(pitch, (k * HOP + WINDOW) / RATE)
        if matcher.last_best is not None:
            best = min(best, matcher.last_best)
    return best


def record_reference(seconds=REFERENCE_SECONDS):
    """Record you whistling the hook, and save it as the reference."""
    audio = record_to(REFERENCE, seconds,
                      "Whistle the first few notes of Waka Waka.")
    template = build_template(REFERENCE)
    describe(audio, template)

    # A reference that can't even recognize its own recording leaves no room
    # for a live attempt, which is never quite identical.
    cost = best_cost_over(REFERENCE, template)
    print(f"\n  self-check: this recording scores {cost:.2f} against itself "
          f"(threshold {MATCH_THRESHOLD})")
    if cost > MATCH_THRESHOLD * 0.6:
        print("         <-- not much margin. A live whistle is never identical,\n"
              "             so it will likely land above the threshold and miss.\n"
              "             Re-record somewhere quieter, holding the notes steady.")
    else:
        print("         (good margin)")
    print(f"\nNow test yourself with:  python {Path(__file__).name} --check")


def check_attempt(template, seconds=REFERENCE_SECONDS):
    """Record one live attempt and report how close it came -- no motor, no
    song, just the number. This is the way to tune MATCH_THRESHOLD."""
    path = ATTEMPT
    audio = record_to(path, seconds, "Whistle it the way you would to set it off.")
    describe(audio, drop_stray_notes(clean_contour(to_array(pitch_track(audio)))))

    cost = best_cost_over(path, template)
    print(f"\n  best match cost: {cost:.2f}   (threshold {MATCH_THRESHOLD})")
    if cost <= MATCH_THRESHOLD:
        print("  -> MATCH. This whistle would have set it off.")
    elif np.isinf(cost):
        print("  -> MISS, and never close: the detector didn't hear enough\n"
              "     whistle to compare. Whistle louder/closer, and check the\n"
              "     'level' line above.")
    else:
        print(f"  -> MISS by {cost - MATCH_THRESHOLD:.2f}.")
        if cost <= MATCH_THRESHOLD + 0.35:
            print(f"     That's close. If several honest attempts land here, raise\n"
                  f"     MATCH_THRESHOLD (near the top of this file) to about "
                  f"{cost + 0.1:.1f}\n     and re-test -- including deliberately"
                  f" whistling the WRONG tune,\n     to check it still says MISS.")
        else:
            print("     That's a long way off -- more likely the reference is the\n"
                  "     problem than the threshold. Re-record it with --record.")
    print(f"\n  (saved as {path.name}; re-run on it any time with "
          f"--test {path.name})")


def run_file(path, matcher):
    """Offline check: stream a recording through the matcher frame by frame."""
    audio = load_audio(path)
    hits = 0
    for k, pitch in enumerate(pitch_track(audio)):
        t = (k * HOP + WINDOW) / RATE
        cost = matcher.update(pitch, t)
        if cost is not None:
            hits += 1
            announce(cost, f"{t:.2f} s")
    print(f"{path}: {hits} match(es)")
    return hits


def run_microphone(matcher, motor, direction, song_path, verbose=False):
    """Listen on the microphone until Ctrl+C.

    sounddevice hands us blocks on its own real-time audio thread; that thread
    does nothing but drop them in a queue. Pitch tracking, matching, Bluetooth
    and playback all happen on the main thread below, so a slow BLE write can
    never stall the microphone."""
    blocks = queue.Queue()

    def audio_callback(indata, frames, time_info, status):
        blocks.put(indata[:, 0].copy())

    reaction = Reaction(motor, direction, song_path)
    tracker = PitchTracker()
    buffer = np.zeros(WINDOW, dtype=np.float32)

    stream = sd.InputStream(samplerate=RATE, channels=1, blocksize=HOP,
                            dtype="float32", callback=audio_callback)
    try:
        with stream:
            def read():
                """Advance by one HOP and return the latest WINDOW samples."""
                nonlocal buffer
                buffer = np.concatenate((buffer[HOP:], blocks.get()))
                return buffer

            print(f"Calibrating: stay quiet for {CALIBRATION_SECONDS:g} s...")
            for _ in range(WINDOW // HOP):  # fill the buffer first
                read()
            n = max(1, int(CALIBRATION_SECONDS * RATE / HOP))
            tracker.calibrate([read().copy() for _ in range(n)])

            with Keyboard() as keys:
                if keys.enabled:
                    print(f"Listening for Waka Waka  "
                          f"('{PAUSE_KEY}' = pause, '{QUIT_KEY}' = quit)\n")
                else:
                    print("Listening for Waka Waka (Ctrl+C to stop)\n")

                paused = False
                while True:
                    key = keys.get()
                    if key == QUIT_KEY:
                        print("\nStopped.")
                        break
                    if key == PAUSE_KEY:
                        paused = not paused
                        if paused:
                            # Stop whatever it's in the middle of, too --
                            # pausing should mean it goes quiet and still.
                            reaction.stop()
                            print(f"\r{'PAUSED -- not listening. Press ' + PAUSE_KEY + ' to resume.':<50}",
                                  end="", flush=True)
                        else:
                            # Forget what was heard across the pause, so half a
                            # tune from before can't combine with half after.
                            matcher.mute_until(time.monotonic())
                            print(f"\r{'Listening again.':<50}", end="", flush=True)

                    # Keep draining the microphone either way: it stops the
                    # queue growing, and keeps the noise floor current so
                    # resuming doesn't need another calibration.
                    pitch = tracker.process(read())
                    if paused:
                        continue

                    now = time.monotonic()
                    reaction.tick(now)

                    cost = matcher.update(pitch, now)
                    status = f"whistle {pitch:5.1f} (MIDI)" if pitch is not None else "..."
                    if verbose and matcher.last_best is not None:
                        status += f"   best cost {matcher.last_best:.2f}"
                    print(f"\r{status:<50}", end="", flush=True)

                    if cost is not None:
                        announce(cost, time.strftime("%H:%M:%S"))
                        ends = reaction.fire(now)
                        # Don't listen to our own speakers while the song plays.
                        matcher.mute_until(ends + COOLDOWN_S)
    except KeyboardInterrupt:
        print("\nStopped.")
    finally:
        reaction.stop()


def connect_motor():
    import legoeducation as le
    from lelib import singleMotor

    motor = singleMotor()
    print(f"Connecting to the single motor ({CARD_COLOR_NAME.lower()} card, {CARD_SERIAL})...")
    motor.connect(card_serial=CARD_SERIAL,
                  card_color=getattr(le, f"LEGO_COLOR_{CARD_COLOR_NAME}"))
    print("Connected.")
    return motor, le.MOTOR_MOVE_DIRECTION_CLOCKWISE


def main():
    parser = argparse.ArgumentParser(
        description="Spin a LEGO motor and play a song when Waka Waka is whistled.")
    parser.add_argument("--record", action="store_true",
                        help="record your reference whistle and exit")
    parser.add_argument("--check", action="store_true",
                        help="record one attempt and report how close it came, "
                             "without the motor or the song")
    parser.add_argument("--test", nargs="+", metavar="FILE",
                        help="run the detector over audio files instead of the mic")
    parser.add_argument("--verbose", action="store_true",
                        help="show the best match cost while whistling (for tuning "
                             "MATCH_THRESHOLD)")
    parser.add_argument("--no-motor", action="store_true",
                        help="skip the Bluetooth connection (audio only)")
    args = parser.parse_args()

    if args.record:
        record_reference()
        return

    if not REFERENCE.exists():
        raise SystemExit(
            f"No reference recording yet ({REFERENCE.name}).\n"
            "Record yourself whistling the hook first:\n"
            "    python waka_motor_control.py --record"
        )

    template = build_template(REFERENCE)
    print(f"Reference: {np.count_nonzero(~np.isnan(template))} whistled frames "
          f"over {len(template) * HOP / RATE:.2f} s")

    if args.check:
        check_attempt(template)
        return

    if args.test:
        for path in args.test:
            run_file(path, TuneMatcher(template))
        return

    song_path = find_song()
    if song_path is None:
        print(f"No song file found -- drop an MP3 in this folder (ideally named "
              f"{SONG_FILE.name}).\nThe motor will still spin for {SPIN_SECONDS:g} s "
              f"on a match.")
    else:
        print(f"Song: {song_path.name}")

    motor, clockwise = (None, None)
    if not args.no_motor:
        motor, clockwise = connect_motor()

    try:
        run_microphone(TuneMatcher(template), motor, clockwise, song_path, args.verbose)
    finally:
        if motor is not None:
            motor.stop()
            motor.disconnect()
            print("Disconnected.")


if __name__ == "__main__":
    main()
