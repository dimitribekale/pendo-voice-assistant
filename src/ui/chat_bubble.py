from PyQt5.QtWidgets import QWidget, QLabel, QVBoxLayout
from PyQt5.QtCore import Qt

class ChatBubble(QWidget):
    def __init__(self, text, is_user=True):
        super().__init__()

        self.layout = QVBoxLayout()
        self.setLayout(self.layout)

        self.label = QLabel(text)
        self.label.setWordWrap(True)
        self.layout.addWidget(self.label)

        if is_user:
            # User's message (outgoing)
            self.setStyleSheet("""
                background-color: #005C4B;
                color: white;
                border-radius: 10px;
                padding: 10px;
            """)
            self.layout.setAlignment(Qt.AlignRight)
        else:
            # Pendo's message (incoming)
            self.setStyleSheet("""
                background-color: #3E3E3E;
                color: white;
                border-radius: 10px;
                padding: 10px;
            """)
            self.layout.setAlignment(Qt.AlignLeft)
