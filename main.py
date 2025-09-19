import sys
from src.gui import MLPVisualizer
from PyQt5.QtWidgets import QApplication

if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = MLPVisualizer()
    window.show()
    sys.exit(app.exec_())