import numpy as np
import pyaudio
import aubio

# Constants
PT_OFFSET = 24.374
PT_SLOPE = 62.511
NUM_INPUT_SAMPLES = 4096
AMPLIFICATION_FACTOR = 2.0

def get_pitch():
    p = pyaudio.PyAudio()

    # Open stream
    stream = p.open(format=pyaudio.paFloat32,
                    channels=1,
                    rate=44100,
                    input=True,
                    frames_per_buffer=NUM_INPUT_SAMPLES)

    # Create pitch detector
    pitch_detector = aubio.pitch("yin", NUM_INPUT_SAMPLES, NUM_INPUT_SAMPLES, 44100)
    pitch_detector.set_unit("Hz")
    pitch_detector.set_tolerance(0.2)

    pitches1 = []

    try:
        while True:
            # Read from audio input stream
            data = stream.read(NUM_INPUT_SAMPLES)
            samples = np.frombuffer(data, dtype=np.float32) * AMPLIFICATION_FACTOR

            # Detect pitch directly on the full buffer
            frequency = pitch_detector(samples)[0]
            #print(f"freq: {frequency}")
            confidence = pitch_detector.get_confidence()

            #print(f"Confidence: {confidence}")

            pitches1.append(frequency if confidence > 0.8 else 0)

            print(f"Pitch1: {pitches1[-1]}")

    except KeyboardInterrupt:
        pass
    finally:
        stream.stop_stream()
        stream.close()
        p.terminate()
    return pitches1

if __name__ == "__main__":
    get_pitch()
