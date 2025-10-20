from PyQt5.QtWidgets import QWidget
from PyQt5.QtGui import QPainter, QColor, QPen, QPainterPath, QLinearGradient
from PyQt5.QtCore import Qt
import numpy as np

class WaveformWidget(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.buffer_size = 2048 # A larger buffer for smoother scrolling
        self.audio_data = np.zeros(self.buffer_size, dtype=np.int16)
        self.setMinimumHeight(150)

    def update_waveform(self, new_audio_data):
        # Roll the buffer and add new data
        self.audio_data = np.roll(self.audio_data, -len(new_audio_data))
        self.audio_data[-len(new_audio_data):] = new_audio_data
        self.update()

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        painter.fillRect(self.rect(), QColor("#2E2E2E"))

        if len(self.audio_data) == 0:
            return

        h = self.height()
        w = self.width()
        center_y = h / 2

        num_samples = len(self.audio_data)
        x_scale = w / num_samples
        y_scale = h / (2 * 32768.0) # Max amplitude for int16

        path = QPainterPath()
        path.moveTo(0, center_y)

        for i in range(num_samples):
            x = int(i * x_scale)
            amplitude = self.audio_data[i]
            y = center_y - (amplitude * y_scale)
            path.lineTo(x, y)
        
        path.lineTo(w, center_y)
        path.lineTo(0, center_y)

        gradient = QLinearGradient(0, 0, 0, h)
        gradient.setColorAt(0.4, QColor("#66FF66"))
        gradient.setColorAt(1.0, QColor("#2E2E2E"))

        painter.setBrush(gradient)
        painter.setPen(Qt.NoPen)
        painter.drawPath(path)
