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


class FaceWindow(QMainWindow, QtStyleTools):
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

        # load widegts form ui file
        self.main = QUiLoader().load('ui.ui', self)
        self.main.button1.clicked.connect(self.video_open_close)
        self.main.button2.clicked.connect(self.register)
        self.main.button3.clicked.connect(self.face_recognition)
        apply_stylesheet(self.main, theme='light_blue.xml')

        self.cap = cv2.VideoCapture(0)
        self.fs = FaceSystem()
        self.face_img = QShowImage()
        self.show()
        self.main.show()
        self.th = threading.Thread(target=self.display)
        self.th.start()
        self.load_feature()
        self.setVisible(False)

        # 建立数据库连接
        self.connection = pymysql.connect(
            host='localhost',
            user='root',
            password='88888888',
            database='face_reconginition',
            charset='utf8mb4',
            cursorclass=pymysql.cursors.DictCursor
        )
        self.cursor = self.connection.cursor()

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
            with self.cursor as cursor:
                # 插入数据的 SQL 语句
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
            self.main.button1.setText('打开摄像头')
            self.main.video.clear()
        else:
            self.open_flag = True
            self.main.button1.setText('关闭摄像头')

    def get_max_boxes(self, faces: np.array):
        areas = (faces[:, 2] - faces[:, 0]) * (faces[:, 3] - faces[:, 1])
        max_index = np.argmax(areas)
        return faces[max_index, :]

    def display(self):
        while self.cap.isOpened():
            if self.open_flag:
                w, h = self.main.video.width(), self.main.video.height()
                # success表示是否读取到帧， frame是读取到的numpy格式图片
                success, frame = self.cap.read()
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
                if self.open_flag:
                    # 将图片显示到QLabel标签中其中main.vedio是一个标签
                    self.main.video.setPixmap(QPixmap.fromImage(img))
                success, frame = self.cap.read()
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

                if self.open_flag:
                    # 将图片显示到QLabel标签中
                    self.main.video.setPixmap(QPixmap.fromImage(img))
                cv2.waitKey(int(1000 / 30))

    def closeEvent(self, event):
        # Set a flag to stop the display thread
        self.open_flag = False

        # Wait for the thread to finish
        if self.th.is_alive():
            self.th.join()

        # Release camera resource
        if self.cap.isOpened():
            self.cap.release()

        # Accept the close event
        super().closeEvent(event)

        # Ensure the application quits
        QApplication.quit()

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
    app = QApplication([])
    window = FaceWindow()
    app.exec()