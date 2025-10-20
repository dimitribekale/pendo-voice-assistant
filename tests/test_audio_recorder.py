import unittest
import os
from src.core.audio_recorder import AudioRecorder

class TestAudioRecorder(unittest.TestCase):
    def test_record_and_save(self):
        recorder = AudioRecorder()
        recorder.start_recording()
        # Simulate recording for a short duration
        for _ in range(10):
            recorder.record_audio()
        
        output_filename = "test_output.wav"
        recorder.stop_recording(output_filename)

        self.assertTrue(os.path.exists(output_filename))
        os.remove(output_filename)

if __name__ == "__main__":
    unittest.main()
