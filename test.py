import threading
import numpy as np
import train
from PySide6 import QtWidgets
from PySide6.QtCore import Qt
from PySide6.QtWidgets import *
from PySide6.QtGui import QImage, QPixmap
from qt_material import apply_stylesheet, QtStyleTools, QUiLoader
from face import *
from PIL import Image
import pymysql
from datetime import datetime





class FaceWindow(QMainWindow, QtStyleTools):
    def __init__(self):
        super().__init__()
        self.show()









    def closeEvent(self, event):
        # Set a flag to stop the display thread
        print("called")

        # Accept the close event
        super().closeEvent(event)

        # Ensure the application quits
        QApplication.quit()




if __name__ == '__main__':
    app = QApplication()
    window = FaceWindow()
    print("created")
    app.exec()