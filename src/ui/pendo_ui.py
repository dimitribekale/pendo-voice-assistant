import sys
import os
import numpy as np
from PyQt5.QtWidgets import QMainWindow, QApplication
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QKeyEvent

# Add the parent directory of 'src' to the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from .pendo_core_widget import PendoCoreWidget
from .context_slate_widget import ContextSlateWidget
from core.audio_recorder import AudioRecorder
from core.speech_transcriber import SpeechTranscriber
from agent.orchestrator import Orchestrator
from core.speech_synthesizer import SpeechSynthesizer

class PendoUI(QMainWindow):
    def __init__(self):
        super().__init__()

        self.setWindowTitle("Pendo")
        # self.setWindowIcon(QIcon("images/pendo-icone.png")) # TODO: Add icon

        self.setWindowFlags(Qt.FramelessWindowHint | Qt.WindowStaysOnTopHint)
        self.setAttribute(Qt.WA_TranslucentBackground)

        screen_geometry = QApplication.primaryScreen().geometry()
        width = 300
        height = 400
        self.setGeometry(screen_geometry.width() - width, screen_geometry.height() - height, width, height)

        self.pendo_core = PendoCoreWidget(self)
        self.setCentralWidget(self.pendo_core)
        self.pendo_core.start_animation()

        self.context_slate = ContextSlateWidget(self)
        self.context_slate.setGeometry(self.width(), 50, 300, 200)
        self.context_slate.hide()
        self.slate_visible = False

        self.setFocusPolicy(Qt.StrongFocus)

        self.audio_recorder = None
        self.speech_transcriber = SpeechTranscriber()
        self.orchestrator = Orchestrator()
        self.speech_synthesizer = SpeechSynthesizer()
        self.is_recording = False

    def keyPressEvent(self, event: QKeyEvent):
        if event.key() == Qt.Key_Space:
            if self.slate_visible:
                self.context_slate.hide_slate()
            else:
                self.context_slate.show_slate()
            self.slate_visible = not self.slate_visible
        elif event.key() == Qt.Key_R:
            self.toggle_recording()
        super().keyPressEvent(event)

    def toggle_recording(self):
        if not self.is_recording:
            self.start_recording()
        else:
            self.stop_recording()

    def start_recording(self):
        self.is_recording = True
        self.pendo_core.stop_animation()
        self.audio_recorder = AudioRecorder()
        self.audio_recorder.data_available.connect(self.update_waveform)
        self.audio_recorder.finished.connect(self.transcribe_audio)
        self.audio_recorder.start_recording()

    def stop_recording(self):
        if self.audio_recorder:
            self.is_recording = False
            self.audio_recorder.stop_recording()
            self.pendo_core.start_animation()
            self.pendo_core.update_audio_level(0)

    def update_waveform(self, data):
        # Calculate RMS of the audio chunk to get a single value for the audio level
        audio_data = np.frombuffer(data, dtype=np.int16)
        rms = np.sqrt(np.mean(audio_data.astype(float)**2))
        # Scale the RMS value to a reasonable range for visualization
        scaled_rms = int(rms / 100)
        self.pendo_core.update_audio_level(scaled_rms)

    def transcribe_audio(self, filename):
        transcribed_text = self.speech_transcriber.transcribe(filename)
        if transcribed_text:
            self.context_slate.label.setText(f"You said: {transcribed_text}")
            if not self.slate_visible:
                self.context_slate.show_slate()
                self.slate_visible = True
            
            response = self.orchestrator.process_command(transcribed_text)
            self.context_slate.label.setText(f"Pendo: {response}")
            self.speech_synthesizer.say(response)

    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton:
            self.drag_position = event.globalPos() - self.frameGeometry().topLeft()
            event.accept()

    def mouseMoveEvent(self, event):
        if event.buttons() == Qt.LeftButton:
            self.move(event.globalPos() - self.drag_position)
            event.accept()


if __name__ == '__main__':

    app = QApplication(sys.argv)
    pendo_ui = PendoUI()
    pendo_ui.show()
    sys.exit(app.exec_())