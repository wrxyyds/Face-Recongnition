import threading
import numpy as np
import train
from PySide6 import QtWidgets
from PySide6.QtCore import Qt, QEvent, Signal
from PySide6.QtWidgets import *
from PySide6.QtGui import QImage, QPixmap
from qt_material import apply_stylesheet, QtStyleTools, QUiLoader
from face import *
from PIL import Image
import pymysql
from datetime import datetime


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


class RegistrationDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("注册信息")

        # 创建输入框
        self.student_id_input = QLineEdit(self)
        self.name_input = QLineEdit(self)
        self.identity_input = QLineEdit(self)

        # 创建标签
        student_id_label = QLabel("学号:", self)
        name_label = QLabel("姓名:", self)
        identity_label = QLabel("身份:", self)

        # 创建按钮
        ok_button = QPushButton("确定", self)
        cancel_button = QPushButton("取消", self)

        # 布局
        layout = QGridLayout()
        layout.addWidget(student_id_label, 0, 0)
        layout.addWidget(self.student_id_input, 0, 1)
        layout.addWidget(name_label, 1, 0)
        layout.addWidget(self.name_input, 1, 1)
        layout.addWidget(identity_label, 2, 0)
        layout.addWidget(self.identity_input, 2, 1)
        layout.addWidget(ok_button, 3, 0)
        layout.addWidget(cancel_button, 3, 1)

        self.setLayout(layout)

        # 连接按钮信号
        ok_button.clicked.connect(self.accept)
        cancel_button.clicked.connect(self.reject)

    def get_inputs(self):
        student_id = self.student_id_input.text()
        name = self.name_input.text()
        identity = self.identity_input.text()
        return student_id, name, identity


class CustomLoader(QUiLoader):
    def __init__(self, base_instance=None):
        super().__init__()
        self.base_instance = base_instance

    def createWidget(self, class_name, parent=None, name=''):
        if parent is None and self.base_instance:
            return self.base_instance
        return super().createWidget(class_name, parent, name)


class FaceWindow(QMainWindow, QtStyleTools):
    shutdown_signal = Signal()  # Define a signal for safe shutdown

    def __init__(self, parent=None):
        super().__init__(parent)
        self.open_flag = False
        self.frame = None
        self.another_frame = None
        self.image = None
        self.faces = None
        self.another_faces = None
        self.features = None
        self.register_ids = None
        self.names = None
        self.identities = None
        self.lock = threading.Lock()  # 添加线程锁
        self.is_shutting_down = False  # Flag to track shutdown state

        # Use CustomLoader to load UI directly into this instance
        loader = CustomLoader(self)
        loader.load('ui.ui')

        # Connect signals
        self.button1.clicked.connect(self.video_open_close)
        self.button2.clicked.connect(self.register)
        self.button3.clicked.connect(self.face_recognition)

        # Apply stylesheet
        apply_stylesheet(self, theme='light_blue.xml')

        # Connect shutdown signal to cleanup method
        self.shutdown_signal.connect(self.cleanup_resources)

        # Connect application quit signal
        QApplication.instance().aboutToQuit.connect(self.on_app_quit)

        self.cap = cv2.VideoCapture(0)
        self.fs = FaceSystem()
        self.face_img = QShowImage()

        # 显示窗口
        self.show()

        # Start display thread
        self.th = threading.Thread(target=self.display)
        self.th.daemon = True  # Make thread daemon so it exits when main thread exits
        self.th.start()
        self.load_feature()

        # 建立数据库连接
        try:
            self.connection = pymysql.connect(
                host='localhost',
                user='root',
                password='88888888',
                database='face_reconginition',
                charset='utf8mb4',
                cursorclass=pymysql.cursors.DictCursor
            )
            self.cursor = self.connection.cursor()
        except pymysql.Error as e:
            print(f"数据库连接错误: {e}")
            self.connection = None
            self.cursor = None
            QtWidgets.QMessageBox.critical(self, '错误', f'数据库连接失败: {e}')

    def save_feature(self, feature, register_id, name, identity):
        # save np feature to file
        id = len(os.listdir('../datas/faces'))
        np.save(f'../datas/faces/{id}.npy', feature)
        with open('../datas/names.txt', 'a', encoding='utf-8') as f:
            f.write(str(register_id))
            f.write(',')
            f.write(name)
            f.write(',')
            f.write(identity)
            f.write('\n')
        if self.features.shape[0] != 0:
            self.features = np.concatenate((self.features, feature), axis=0)
        else:
            self.features = feature.reshape(1, -1)
        self.register_ids.append(register_id)
        self.names.append(name)
        self.identities.append(identity)

    def load_feature(self):
        self.features = []
        self.register_ids = []
        self.names = []
        self.identities = []
        feature_files = os.listdir('../datas/faces')
        if len(feature_files) > 0:
            for file in feature_files:
                self.features.append(np.load(f'../datas/faces/{file}'))
            with open('../datas/names.txt', 'r') as f:
                for line in f.readlines():
                    line = line.strip()
                    register_id, name, identity = line.split(',')
                    self.register_ids.append(register_id)
                    self.names.append(name)
                    self.identities.append(identity)
            self.features = np.array(self.features)
            self.features = np.squeeze(self.features)
        else:
            self.features = np.empty((0, 0))

    def calculate_distance(self, feature):
        # self.features的形状是(num, batch_size = 1, 512)
        distance = np.sqrt(np.sum(np.square(self.features - feature), axis=1))
        min_index = np.argmin(distance)
        return min_index, distance[min_index]

    def face_recognition(self):
        with self.lock:  # 使用线程锁
            if self.faces is None or self.faces.shape[0] == 0:
                QtWidgets.QMessageBox.warning(self, '提示', f'未识别到人脸')
                return

        face = np.ascontiguousarray(
            self.frame[self.faces[0][1]:self.faces[0][3], self.faces[0][0]:self.faces[0][2]])
        feature = self.fs.get_face_feature(Image.fromarray(face))
        min_index, min_dis = self.calculate_distance(feature)
        print('min_dis', min_dis)

        if min_dis < 0.5:
            now = datetime.now()
            punch_time = now.strftime('%Y-%m-%d %H:%M:%S')

            if not self.connection:
                QtWidgets.QMessageBox.warning(self, '错误', '数据库连接不可用')
                return

            cursor = None
            try:
                # 插入数据的 SQL 语句
                cursor = self.connection.cursor()
                if self.identities[min_index] == 'student':
                    sql = "INSERT INTO student_attendance (student_id, student_name, punch_time) VALUES (%s, %s, %s)"
                    values = (str(self.register_ids[min_index]), str(self.names[min_index]), punch_time)
                else:
                    sql = "INSERT INTO teacher_attendance (teacher_id, teacher_name, punch_time) VALUES (%s, %s, %s)"
                    values = (str(self.register_ids[min_index]), str(self.names[min_index]), punch_time)

                # 执行插入操作
                cursor.execute(sql, values)

                # 提交事务
                self.connection.commit()
                print("数据插入成功！")

                QtWidgets.QMessageBox.information(self, '提示', f'人脸识别结果为{self.register_ids[min_index]}')
            except Exception as e:
                # 发生错误时回滚
                if self.connection:
                    self.connection.rollback()
                print(f"数据库错误: {e}")
                QtWidgets.QMessageBox.warning(self, '错误', f'数据库操作失败: {e}')
            finally:
                # 确保手动关闭游标
                if cursor:
                    cursor.close()
        else:
            QtWidgets.QMessageBox.information(self, '提示', '人脸不在数据库中')

    def register(self):
        with self.lock:  # 使用线程锁
            # 弹窗提示 先打开摄像头
            if self.faces is None:
                QApplication.setQuitOnLastWindowClosed(False)
                QtWidgets.QMessageBox.warning(self, '提示', '请先打开摄像头并确保画面中有人脸')
                return

        face1 = np.ascontiguousarray(
            self.frame[self.faces[0][1]:self.faces[0][3], self.faces[0][0]:self.faces[0][2]])
        face2 = np.ascontiguousarray(
            self.another_frame[self.another_faces[0][1]:self.another_faces[0][3],
            self.another_faces[0][0]:self.another_faces[0][2]])
        reply1 = QtWidgets.QMessageBox.question(self, '提示', '是否要使用该图像进行特征提取',
                                                QMessageBox.No | QMessageBox.Yes)
        if reply1 == QMessageBox.StandardButton.Yes:
            feature = self.fs.get_face_feature(Image.fromarray(face1))
            if self.features.shape[0] != 0:
                min_index, min_dis = self.calculate_distance(feature)
                print(min_dis)
                if min_dis < 0.5:
                    QtWidgets.QMessageBox.warning(self, '提示', f'该人脸已经注册过!')
                else:
                    dialog = RegistrationDialog(self)
                    if dialog.exec_() == QDialog.Accepted:
                        register_id, name, identity = dialog.get_inputs()
                        if name:
                            self.save_image(face1, '../images/train/' + str(register_id) + '/' + 'face_0.jpg')
                            self.save_image(face2, '../images/train/' + str(register_id) + '/' + 'face_1.jpg')
                            train.train(str(register_id))
                            self.fs = FaceSystem()
                            feature = self.fs.get_face_feature(Image.fromarray(face1))
                            self.save_feature(feature, str(register_id), str(name), str(identity))
                            QtWidgets.QMessageBox.information(self, '提示', f'人脸注册成功！')
            else:
                dialog = RegistrationDialog(self)
                if dialog.exec_() == QDialog.Accepted:
                    register_id, name, identity = dialog.get_inputs()
                    if name:
                        self.save_image(face1, '../images/train/' + str(register_id) + '/' + 'face_0.jpg')
                        self.save_image(face2, '../images/train/' + str(register_id) + '/' + 'face_1.jpg')
                        train.train(str(register_id))
                        self.fs = FaceSystem()
                        feature = self.fs.get_face_feature(Image.fromarray(face1))
                        self.save_feature(feature, str(register_id), str(name), str(identity))
                        QtWidgets.QMessageBox.information(self, '提示', f'人脸注册成功！')

    def video_open_close(self):
        if self.open_flag:
            self.open_flag = False
            self.button1.setText('打开摄像头')
            self.video.clear()
        else:
            self.open_flag = True
            self.button1.setText('关闭摄像头')

    def get_max_boxes(self, faces: np.array):
        areas = (faces[:, 2] - faces[:, 0]) * (faces[:, 3] - faces[:, 1])
        max_index = np.argmax(areas)
        return faces[max_index, :]

    def display(self):
        try:
            while self.cap.isOpened() and not self.is_shutting_down:
                if self.open_flag:
                    try:
                        w, h = self.video.width(), self.video.height()
                        # success表示是否读取到帧， frame是读取到的numpy格式图片
                        success, frame = self.cap.read()
                        if not success:
                            print("Failed to read frame from camera")
                            time.sleep(0.1)
                            continue

                        frame = cv2.resize(frame, (w - 20, h - 20), interpolation=cv2.INTER_AREA)
                        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                        image = Image.fromarray(frame, mode='RGB')
                        # frame大小是(w - 20, h - 20)给组件预留空白
                        frame = Image.fromarray(frame, mode='RGB')
                        # 重新设置格式
                        image = image.resize((512, 512))
                        # mtcnn进行人脸检测返回一个二维numpy第0维度是数量，第一维度是四个坐标
                        faces = self.fs.face_detect(image)
                        faces = np.array(faces)
                        if faces.shape[0] > 0:
                            # 选择面积最大的box，此时faces是一维的
                            faces = self.get_max_boxes(faces)
                            # 添加一个新的维度，归一化。以frame的尺度计算得到人脸坐标
                            faces = faces[np.newaxis, :]
                            faces = faces / 512
                            faces[:, 0] = faces[:, 0] * (w - 20)
                            faces[:, 1] = faces[:, 1] * (h - 20)
                            faces[:, 2] = faces[:, 2] * (w - 20)
                            faces[:, 3] = faces[:, 3] * (h - 20)
                            faces = faces.astype(int)
                            # 将人脸坐标画出来
                            frame = draw_bboxes(frame, faces)
                            with self.lock:  # 使用线程锁
                                # 将faces在内存空间连续排列
                                self.faces = np.ascontiguousarray(faces)
                        self.frame = np.ascontiguousarray(np.array(frame))
                        # 将PIL格式转为QImage self.frame.data返回存储数据指针，在使用前确保连续存储
                        img = QImage(self.frame.data, self.frame.shape[1], self.frame.shape[0], self.frame.shape[1] * 3,
                                     QImage.Format_RGB888)
                        if self.open_flag and not self.is_shutting_down:
                            # 将图片显示到QLabel标签中其中video是一个标签
                            self.video.setPixmap(QPixmap.fromImage(img))

                        success, frame = self.cap.read()
                        if not success:
                            continue

                        frame = cv2.resize(frame, (w - 20, h - 20), interpolation=cv2.INTER_AREA)
                        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                        image = Image.fromarray(frame, mode='RGB')
                        # frame大小是(w - 20, h - 20)给组件预留空白
                        frame = Image.fromarray(frame, mode='RGB')
                        # 重新设置格式
                        image = image.resize((512, 512))
                        # mtcnn进行人脸检测返回一个二维numpy第0维度是数量，第一维度是四个坐标
                        faces = self.fs.face_detect(image)
                        faces = np.array(faces)
                        if faces.shape[0] > 0:
                            # 选择面积最大的box，此时faces是一维的
                            faces = self.get_max_boxes(faces)
                            # 添加一个新的维度，归一化。以frame的尺度计算得到人脸坐标
                            faces = faces[np.newaxis, :]
                            faces = faces / 512
                            faces[:, 0] = faces[:, 0] * (w - 20)
                            faces[:, 1] = faces[:, 1] * (h - 20)
                            faces[:, 2] = faces[:, 2] * (w - 20)
                            faces[:, 3] = faces[:, 3] * (h - 20)
                            faces = faces.astype(int)
                            # 将人脸坐标画出来
                            frame = draw_bboxes(frame, faces)
                            with self.lock:  # 使用线程锁
                                # 将faces在内存空间连续排列
                                self.another_faces = np.ascontiguousarray(faces)
                        self.another_frame = np.ascontiguousarray(np.array(frame))

                        # 使用当前帧的数据创建QImage，而不是self.frame
                        img = QImage(self.another_frame.data, self.another_frame.shape[1], self.another_frame.shape[0],
                                     self.another_frame.shape[1] * 3, QImage.Format_RGB888)

                        if self.open_flag and not self.is_shutting_down:
                            # 将图片显示到QLabel标签中
                            self.video.setPixmap(QPixmap.fromImage(img))

                        cv2.waitKey(int(1000 / 30))

                    except Exception as e:
                        print(f"Error in display loop: {e}")
                        time.sleep(0.1)
                else:
                    # Avoid high CPU usage when camera is closed
                    time.sleep(0.1)
        except Exception as e:
            print(f"Display thread crashed: {e}")
        finally:
            print("Display thread terminated")

    def closeEvent(self, event):
        """Handle window close event"""
        print("FaceWindow closeEvent triggered")
        # Prevent multiple closeEvent handling
        if not self.is_shutting_down:
            self.is_shutting_down = True
            self.shutdown_signal.emit()
            # Accept the event after cleanup signal is emitted
            event.accept()
        else:
            # Already shutting down, just accept the event
            event.accept()

    def on_app_quit(self):
        """Handle application quit signal"""
        print("Application quit signal received")
        if not self.is_shutting_down:
            self.is_shutting_down = True
            self.cleanup_resources()

    def cleanup_resources(self):
        """清理所有资源的函数"""
        print("Cleaning up resources...")

        # 设置标志停止显示线程
        self.open_flag = False
        self.is_shutting_down = True

        # 等待线程结束，使用较短的超时防止无限等待
        if hasattr(self, 'th') and self.th.is_alive():
            print("Waiting for display thread to finish...")
            # Shorter timeout to avoid blocking UI for too long
            self.th.join(timeout=0.5)
            print("Display thread finished or timeout reached")

        # 释放摄像头资源
        if hasattr(self, 'cap') and self.cap.isOpened():
            print("Releasing camera...")
            self.cap.release()
            print("Camera released")

        # 关闭数据库连接
        if hasattr(self, 'connection') and self.connection:
            print("Closing database connection...")
            if hasattr(self, 'cursor') and self.cursor:
                self.cursor.close()
            self.connection.close()
            print("Database connection closed")

    def save_image(self, image, path):
        """
        将图片保存到指定路径
        :param image: 要保存的图片（numpy数组）
        :param path: 保存的路径
        """
        # 确保保存路径的目录存在
        directory = os.path.dirname(path)
        if not os.path.exists(directory):
            os.makedirs(directory)

        # 将numpy数组转换为BGR格式
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        # 保存图片
        cv2.imwrite(path, image)


if __name__ == '__main__':
    import sys
    import time


    # Add exception hook to catch unhandled exceptions
    def exception_hook(exctype, value, traceback):
        print(f"Unhandled exception: {exctype.__name__}: {value}")
        sys.__excepthook__(exctype, value, traceback)


    sys.excepthook = exception_hook

    app = QApplication(sys.argv)
    window = FaceWindow()
    sys.exit(app.exec())