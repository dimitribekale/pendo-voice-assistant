import sys
from pathlib import Path
from PyQt5.QtWidgets import QWidget, QApplication
from PyQt5.QtGui import QPainter, QPixmap, QColor, QBrush, QRadialGradient, QPainterPath
from PyQt5.QtCore import Qt, QTimer, QPropertyAnimation, pyqtProperty, QEasingCurve

current_file = Path(__file__).resolve()
project_root = current_file.parent.parent.parent
icon_path = project_root / 'assets' / 'images' / 'pendo-icone.png'

class PendoCoreWidget(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMinimumSize(200, 200)

        self.pendo_icon = QPixmap(str(icon_path))
        self.icon_size = 80
        self.pendo_icon = self.pendo_icon.scaled(self.icon_size, self.icon_size, Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation)

        self._radius = 50 # Base radius for the core
        self._pulse_radius = 0
        self._audio_level = 0

        # Define the animation first before referencing it
        self.pulse_animation = QPropertyAnimation(self, b"pulse_radius")
        self.pulse_animation.setDuration(1500)
        self.pulse_animation.setStartValue(0)
        self.pulse_animation.setEndValue(20)
        self.pulse_animation.setEasingCurve(QEasingCurve.InOutQuad)
        self.pulse_animation.setLoopCount(-1) # Loop indefinitely

    @pyqtProperty(int)
    def pulse_radius(self):
        return self._pulse_radius

    @pulse_radius.setter
    def pulse_radius(self, value):
        self._pulse_radius = value
        self.update() # Trigger a repaint

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)

        width = self.width()
        height = self.height()
        center_x = width // 2
        center_y = height // 2

        # Draw the corona (outer glow) - uses _pulse_radius and _audio_level
        corona_radius = self._radius + self._pulse_radius + self._audio_level
        corona_gradient = QRadialGradient(center_x, center_y, corona_radius)
        corona_gradient.setColorAt(0, QColor(248, 231, 28, 150))
        corona_gradient.setColorAt(0.8, QColor(248, 231, 28, 50))
        corona_gradient.setColorAt(1, QColor(248, 231, 28, 0))
        painter.setBrush(QBrush(corona_gradient))
        painter.setPen(Qt.PenStyle.NoPen)
        painter.drawEllipse(center_x - corona_radius, center_y - corona_radius, corona_radius * 2, corona_radius * 2)

        # Draw the inner core (blue orb) - this will be behind the icon
        core_gradient = QRadialGradient(center_x, center_y, self._radius)
        core_gradient.setColorAt(0, QColor(74, 144, 226, 255))
        core_gradient.setColorAt(0.8, QColor(74, 144, 226, 200))
        core_gradient.setColorAt(1, QColor(74, 144, 226, 100))
        painter.setBrush(QBrush(core_gradient))
        painter.drawEllipse(center_x - self._radius, center_y - self._radius, self._radius * 2, self._radius * 2)

        # Draw the pendo icon, clipped to a circle
        icon_rect_x = center_x - self.pendo_icon.width() // 2
        icon_rect_y = center_y - self.pendo_icon.height() // 2
        icon_rect_width = self.pendo_icon.width()
        icon_rect_height = self.pendo_icon.height()

        path = QPainterPath()
        path.addEllipse(icon_rect_x, icon_rect_y, icon_rect_width, icon_rect_height)
        painter.setClipPath(path)
        painter.drawPixmap(icon_rect_x, icon_rect_y, self.pendo_icon)
        painter.setClipping(False) # Reset clipping

    def update_audio_level(self, level):
        self._audio_level = level
        self.update() # Trigger a repaint


    def start_animation(self):
        self.pulse_animation.start()

    def stop_animation(self):
        self.pulse_animation.stop()

if __name__ == '__main__':
    app = QApplication(sys.argv)
    widget = PendoCoreWidget()
    widget.show()
    widget.start_animation()
    sys.exit(app.exec_())