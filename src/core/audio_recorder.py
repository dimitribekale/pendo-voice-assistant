import pyaudio
import wave
from PyQt5.QtCore import QThread, pyqtSignal
import numpy as np

class AudioRecorder(QThread):
    finished = pyqtSignal(str)
    data_available = pyqtSignal(np.ndarray)

    def __init__(self, chunk=1024, sample_format=pyaudio.paInt16, channels=1, fs=44100):
        super().__init__()
        self.chunk = chunk
        self.sample_format = sample_format
        self.channels = channels
        self.fs = fs
        self.frames = []
        self.p = None
        self.stream = None
        self.is_recording = False
        self.output_filename = "output.wav"

    def run(self):
        self.p = pyaudio.PyAudio()
        self.stream = self.p.open(format=self.sample_format,
                                  channels=self.channels,
                                  rate=self.fs,
                                  frames_per_buffer=self.chunk,
                                  input=True)
        self.is_recording = True
        self.frames = []
        print("Recording...")

        while self.is_recording:
            data = self.stream.read(self.chunk)
            self.frames.append(data)
            audio_data = np.frombuffer(data, dtype=np.int16)
            self.data_available.emit(audio_data)

        print("Finished recording.")
        self.stream.stop_stream()
        self.stream.close()
        self.p.terminate()

        wf = wave.open(self.output_filename, 'wb')
        wf.setnchannels(self.channels)
        wf.setsampwidth(self.p.get_sample_size(self.sample_format))
        wf.setframerate(self.fs)
        wf.writeframes(b''.join(self.frames))
        wf.close()
        self.finished.emit(self.output_filename)

    def start_recording(self):
        self.start()

    def stop_recording(self, filename="output.wav"):
        self.output_filename = filename
        self.is_recording = False
