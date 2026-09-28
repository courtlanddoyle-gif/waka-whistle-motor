"""
Control a single motor (purple card, serial 6235) with whistle pitch:
    high pitch   -> spin clockwise
    low pitch    -> spin counterclockwise
    middle pitch -> rapidly alternate between both directions
    silence      -> stop

Listens on the system default microphone through a sounddevice InputStream,
and picks the pitch out of each block with a plain FFT peak: whichever
frequency is loudest wins. That's the whole detector -- there's no melody
matching here, unlike waka_motor_control.py in this folder.

LOW_PITCH_MAX_HZ / HIGH_PITCH_MIN_HZ below are placeholders -- tune them to
your own whistle range. Running `python waka_motor_control.py --check` reports
the frequency range it hears you whistling in; set them from that.

Run:
    python whistle_motor_control.py
Press Ctrl+C to quit.
"""

import threading
import time

import numpy as np
import sounddevice as sd

import legoeducation as le
from lelib import singleMotor

CARD_COLOR = le.LEGO_COLOR_PURPLE
CARD_SERIAL = 6235

SAMPLE_RATE = 44100
BLOCK_SIZE = 2048

# TODO: tune these to your actual whistle range.
LOW_PITCH_MAX_HZ = 800     # at or below this -> "low"
HIGH_PITCH_MIN_HZ = 1600   # at or above this -> "high"; between the two -> "middle"
MIN_AMPLITUDE = 0.02       # ignore quiet background noise (0-1 scale)

MOTOR_SPEED = 60           # percent, 0-100
OSCILLATE_INTERVAL_S = 0.2 # how often to flip direction while whistling "middle"
POLL_DELAY_S = 0.05        # how often the main loop checks the detected pitch zone

# Shared between the audio callback thread and the main thread.
_lock = threading.Lock()
_current_zone = "silence"  # "low", "middle", "high", or "silence"


def dominant_frequency(block, sample_rate):
    """Return the loudest frequency (Hz) in this audio block, or None if
    too quiet to trust (background noise)."""
    if np.sqrt(np.mean(block ** 2)) < MIN_AMPLITUDE:
        return None
    windowed = block * np.hanning(len(block))
    spectrum = np.abs(np.fft.rfft(windowed))
    freqs = np.fft.rfftfreq(len(block), d=1.0 / sample_rate)
    return freqs[np.argmax(spectrum)]


def classify_pitch(freq):
    if freq is None:
        return "silence"
    if freq <= LOW_PITCH_MAX_HZ:
        return "low"
    if freq >= HIGH_PITCH_MIN_HZ:
        return "high"
    return "middle"


def audio_callback(indata, frames, time_info, status):
    """Runs on sounddevice's real-time audio thread -- kept fast (just a
    classification + a variable write), with actual motor commands issued
    from the main loop instead of here."""
    global _current_zone
    zone = classify_pitch(dominant_frequency(indata[:, 0], SAMPLE_RATE))
    with _lock:
        _current_zone = zone


def main():
    device_index = None  # always use the system default microphone

    motor = singleMotor()
    print("Connecting to single motor...")
    motor.connect(card_serial=CARD_SERIAL, card_color=CARD_COLOR)
    print("Connected.")

    oscillate_clockwise = False
    last_oscillate_time = 0.0
    current_action = None  # avoids re-sending the same BLE command every loop

    try:
        with sd.InputStream(samplerate=SAMPLE_RATE, channels=1,
                             blocksize=BLOCK_SIZE, dtype="float32",
                             device=device_index, callback=audio_callback):
            print("Listening. Whistle high/low/middle to control the motor. Ctrl+C to stop.\n")

            while True:
                with _lock:
                    zone = _current_zone
                now = time.monotonic()

                if zone == "high":
                    if current_action != "high":
                        motor.motor_run(direction=le.MOTOR_MOVE_DIRECTION_CLOCKWISE, speed=MOTOR_SPEED)
                        current_action = "high"
                        print("HIGH -- clockwise")

                elif zone == "low":
                    if current_action != "low":
                        motor.motor_run(direction=le.MOTOR_MOVE_DIRECTION_COUNTERCLOCKWISE, speed=MOTOR_SPEED)
                        current_action = "low"
                        print("LOW -- counterclockwise")

                elif zone == "middle":
                    if now - last_oscillate_time >= OSCILLATE_INTERVAL_S:
                        oscillate_clockwise = not oscillate_clockwise
                        direction = (le.MOTOR_MOVE_DIRECTION_CLOCKWISE if oscillate_clockwise
                                     else le.MOTOR_MOVE_DIRECTION_COUNTERCLOCKWISE)
                        motor.motor_run(direction=direction, speed=MOTOR_SPEED)
                        last_oscillate_time = now
                    current_action = "middle"

                else:  # silence
                    if current_action != "silence":
                        motor.motor_stop()
                        current_action = "silence"
                        print("SILENCE -- stop")

                time.sleep(POLL_DELAY_S)
    except (KeyboardInterrupt, EOFError):
        print("\nStopping...")
    finally:
        motor.stop()
        motor.disconnect()
        print("Disconnected.")


if __name__ == "__main__":
    main()
