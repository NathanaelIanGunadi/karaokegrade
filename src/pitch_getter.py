import numpy as np
import pyaudio
import aubio

# Constants
PT_OFFSET = 24.374
PT_SLOPE = 62.511
NUM_INPUT_SAMPLES = 1 << 12

def get_pitch_from_stream():
    p = pyaudio.PyAudio()

    # Open stream
    stream = p.open(format=pyaudio.paFloat32,
                    channels=1,
                    rate=44100,
                    input=True,
                    frames_per_buffer=NUM_INPUT_SAMPLES)

    # Create pitch detector
    pitch_detector = aubio.pitch("default", NUM_INPUT_SAMPLES, NUM_INPUT_SAMPLES // 8, 44100)
    pitch_detector.set_unit("Hz")
    pitch_detector.set_tolerance(0.2)

    pitches1 = []
    pitches2 = []

    try:
        while True:
            # Read from audio input stream
            data = stream.read(NUM_INPUT_SAMPLES)
            samples = np.frombuffer(data, dtype=np.float32)

            # Detect pitch
            frequency = pitch_detector(samples)[0]
            confidence = pitch_detector.get_confidence()

            # This part is simplistic, but you can modify it to handle stereo channels
            # if your setup requires it.
            pitches1.append(frequency if confidence > 0.8 else 0)
            pitches2.append(frequency if confidence > 0.8 else 0)

            print(f"Pitch1: {pitches1[-1]}, Pitch2: {pitches2[-1]}")
    except KeyboardInterrupt:
        pass
    finally:
        stream.stop_stream()
        stream.close()
        p.terminate()
    return pitches1, pitches2

if __name__ == "__main__":
    get_pitch_from_stream()