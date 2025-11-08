import wave
import logging
import pyaudio
import numpy as np
from PyQt5.QtCore import QThread, pyqtSignal


logger = logging.getLogger(__name__)

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
        self.output_filename = "output.wav"

    def run(self):
        try:
            logger.info("Initializing audio device...")
            self.p = pyaudio.PyAudio()

            try: # Open audio stream
                self.stream = self.p.open(
                    format=self.sample_format,
                    channels=self.channels,
                    rate=self.fs,
                    frames_per_buffer=self.chunk,
                    input=True
                )
            except OSError as e:
                logger.error(f"Failed to open audio stream: {e}")
                logger.error("Check that the microphone is available.")
                self.p.terminate()
                self.finished.emit("")
                return
            
            self.frames = []
            logger.info("Recording started...")

            # Record audio loop (thread-safe using Qt's interruption mechanism)
            while not self.isInterruptionRequested():
                try:
                    data = self.stream.read(self.chunk, exception_on_overflow=False)
                    self.frames.append(data)
                    audio_data = np.frombuffer(data, dtype=np.int16)
                    self.data_available.emit(audio_data)
                except OSError as e:
                    logger.error(f"Error reading audio stream: {e}")
                    break
            logger.info("Recording finished.")

            # Cleanup audio resources
            try:
                if self.stream:
                    self.stream.stop_stream()
                    self.stream.close()
                if self.p:
                    self.p.terminate()
            except Exception as e:
                logger.warning(f"Error during audio cleanup: {e}")
            
            # Save the recorded audio to file
            if not self.frames:
                logger.warning("No audio data recorded")
                self.finished.emit("")
                return
            
            try:
                logger.info(f"Saving audio to {self.output_filename}...")
                wf = wave.open(self.output_filename, 'wb')
                wf.setnchannels(self.channels)
                wf.setsampwidth(self.p.get_sample_size(self.sample_format))
                wf.setframerate(self.fs)
                wf.writeframes(b''.join(self.frames))
                wf.close()
                logger.info(f"Audio saved successfully to {self.output_filename}")
                self.finished.emit(self.output_filename)

            except IOError as e:
                logger.exception(f"Unexpected error saving audio: {e}")
                self.finished.emit("")

        except Exception as e:
            # Catch-all for any unexpected errors
            logger.exception(f"Unexpected error in audio recording: {e}")
            # Attempt cleanup
            try:
                if hasattr(self, 'stream') and self.stream:
                    self.stream.close()
                if hasattr(self, 'p') and self.p:
                    self.p.terminate()
            except:
                pass
            self.finished.emit("")

    def start_recording(self):
        self.start()

    def stop_recording(self, filename="output.wav"):
        """
        Stops the recording thread safely.

        Uses Qt's thread-safe requestInterruption() mechanism instead of
        a shared boolean variable to avoid race conditions.

        Args:
            filename: Output filename for the recorded audio
        """
        self.output_filename = filename
        self.requestInterruption()  # Thread-safe way to signal the recording thread to stop
        logger.info("Stop recording requested")
