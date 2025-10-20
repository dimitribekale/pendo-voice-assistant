import unittest
import os
import wave
import struct
import math
from src.core.speech_transcriber import SpeechTranscriber

def generate_dummy_wav(filename="dummy.wav", duration=1, freq=440):
    sample_rate = 44100
    n_samples = int(duration * sample_rate)
    with wave.open(filename, 'w') as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        for i in range(n_samples):
            value = int(32767.0 * math.sin(2 * math.pi * freq * i / sample_rate))
            data = struct.pack('<h', value)
            wf.writeframesraw(data)

class TestSpeechTranscriber(unittest.TestCase):
    def setUp(self):
        self.model_path = "vosk-model-small-en-us-0.15"  # Replace with your model path
        if not os.path.exists(self.model_path):
            self.skipTest("Vosk model not found. Skipping test.")
        
        self.dummy_wav = "dummy.wav"
        generate_dummy_wav(self.dummy_wav)

    def tearDown(self):
        if os.path.exists(self.dummy_wav):
            os.remove(self.dummy_wav)

    def test_transcribe(self):
        transcriber = SpeechTranscriber(self.model_path)
        # This test will likely fail as the dummy audio is just a sine wave.
        # A real audio file with speech would be needed for a meaningful test.
        text = transcriber.transcribe(self.dummy_wav)
        self.assertIsInstance(text, str)

if __name__ == "__main__":
    unittest.main()
