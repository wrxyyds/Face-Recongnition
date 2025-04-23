import sys
from PySide6 import QtGui, QtWidgets
from PySide6.QtGui import QImage, QPixmap
from PySide6.QtWidgets import QApplication, QWidget, QLabel, QHBoxLayout


class QShowImage(QWidget):
    def __init__(self):
        super(QShowImage, self).__init__()
        self.label = QLabel(self)
        self.layout = QHBoxLayout()
        self.layout.addWidget(self.label)
        self.setLayout(self.layout)
        # 设置窗口标题
        self.setWindowTitle("Image")

    def set_image(self, q_image):
        pixmap = QPixmap.fromImage(q_image)
        self.label.setPixmap(pixmap)


if __name__ == '__main__':
    app = QApplication(sys.argv)

    # 创建 QShowImage 实例
    show_image_widget = QShowImage()

    # 假设我们有一个简单的测试图像（这里创建一个 200x200 的红色图像作为示例）
    width = 200
    height = 200
    bytes_per_line = 3 * width
    image = QImage(width, height, QImage.Format_RGB888)
    for y in range(height):
        for x in range(width):
            image.setPixelColor(x, y, QtGui.QColor(255, 0, 0))

    # 设置图像到 QShowImage 小部件
    show_image_widget.set_image(image)

    # 显示窗口
    show_image_widget.show()

    sys.exit(app.exec())