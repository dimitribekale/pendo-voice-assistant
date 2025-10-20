import sys
import os
os.environ["CUDA_VISIBLE_DEVICES"] = ""
import torch
from PyQt5.QtWidgets import QApplication
from ui.pendo_ui import PendoUI



def main():
    app = QApplication(sys.argv)
    pendo_ui = PendoUI()
    pendo_ui.show()
    sys.exit(app.exec_())

if __name__ == "__main__":
    main()