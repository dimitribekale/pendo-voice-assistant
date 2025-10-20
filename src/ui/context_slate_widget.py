import sys
from PyQt5.QtWidgets import QWidget, QApplication, QVBoxLayout, QLabel
from PyQt5.QtGui import QColor
from PyQt5.QtCore import Qt, QPropertyAnimation, QRect, QEasingCurve, QTimer

class ContextSlateWidget(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)

        self.setWindowFlags(Qt.WindowType.FramelessWindowHint | Qt.WindowType.Tool)
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground)
        self.setAttribute(Qt.WidgetAttribute.WA_StyledBackground, True)

        # TODO: Implement blur effect for glassmorphism
        self.setStyleSheet("background-color: rgba(46, 46, 46, 0.8); border-radius: 10px;")

        self.layout = QVBoxLayout(self)
        self.layout.setContentsMargins(20, 20, 20, 20)

        # Placeholder for content cards
        self.label = QLabel("Context Slate")
        self.label.setStyleSheet("color: white; font-size: 16px;")
        self.layout.addWidget(self.label)

        self.animation = QPropertyAnimation(self, b"geometry")

    def show_slate(self):
        self.show()
        start_geometry = self.geometry()
        end_geometry = self.geometry()
        end_geometry.moveLeft(self.parent().width() - self.width() - 20)

        self.animation.setDuration(300)
        self.animation.setStartValue(start_geometry)
        self.animation.setEndValue(end_geometry)
        self.animation.setEasingCurve(QEasingCurve.InOutQuad)
        self.animation.start()

    def hide_slate(self):
        start_geometry = self.geometry()
        end_geometry = self.geometry()
        end_geometry.moveLeft(self.parent().width())

        self.animation.setDuration(300)
        self.animation.setStartValue(start_geometry)
        self.animation.setEndValue(end_geometry)
        self.animation.setEasingCurve(QEasingCurve.InOutQuad)
        self.animation.finished.connect(self.hide)
        self.animation.start()

if __name__ == '__main__':
    app = QApplication(sys.argv)

    # Create a main window to test the context slate
    main_window = QWidget()
    main_window.setFixedSize(400, 600)

    slate = ContextSlateWidget(main_window)
    slate.setGeometry(main_window.width(), 50, 300, 200)

    # Show the slate after a delay
    QTimer.singleShot(1000, slate.show_slate)
    # Hide the slate after another delay
    QTimer.singleShot(4000, slate.hide_slate)

    main_window.show()
    sys.exit(app.exec_())
