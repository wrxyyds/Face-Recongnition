import threading


import torch
from PySide6 import QtWidgets
from PySide6.QtCore import Qt
from PySide6.QtWidgets import *
from PySide6.QtGui import QImage, QPixmap
from qt_material import apply_stylesheet, QtStyleTools, QUiLoader
from torch.utils.data import DataLoader
from torchvision import transforms
from torch.optim.lr_scheduler import StepLR
import torch.optim as optim
import torch.nn as nn

from face import *
from PIL import Image
from face.facenet.model import Resnet34Triplet
from train import CustomFaceDataset, train_step


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



class FaceWindow(QMainWindow, QtStyleTools):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.open_flag = False
        self.frame = None
        self.image = None
        self.faces = None
        self.features = None
        self.names = None

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

    def save_feature(self, feature, name):
        # save np feature to file
        id = len(os.listdir('../datas/faces'))
        np.save(f'../datas/faces/{id}.npy', feature)
        with open('../datas/names.txt', 'a') as f:
            f.write(f'{name}\n')
        if self.features.shape[0] != 0:
            self.features = np.concatenate((self.features, feature), axis=0)
        else:
            self.features = feature.reshape(1, -1)
        self.names.append(name)

    def load_feature(self):
        self.features = []
        self.names = []
        feature_files = os.listdir('../datas/faces')
        if len(feature_files) > 0:
            for file in feature_files:
                self.features.append(np.load(f'../datas/faces/{file}'))
            with open('../datas/names.txt', 'r') as f:
                for line in f.readlines():
                    self.names.append(line.strip())
            self.features = np.array(self.features)
        else:
            self.features = np.empty((0, 0))

    def calculate_distance(self, feature):
        distance = np.sqrt(np.sum(np.square(self.features - feature),axis=1))
        min_index = np.argmin(distance)
        return self.names[min_index], distance[min_index]

    def face_recognition(self):
        if self.faces.shape[0] == 0:
            QtWidgets.QMessageBox.warning(self, '提示', f'未识别到人脸')
            return

        face = np.ascontiguousarray(
            self.frame[self.faces[0][1]:self.faces[0][3], self.faces[0][0]:self.faces[0][2]])
        #self.face_img.set_image(QImage(face.data, face.shape[1], face.shape[0], face.shape[1] * 3,
                                       #QImage.Format_RGB888))
        #self.face_img.setVisible(True)
        feature = self.fs.get_face_feature(Image.fromarray(face))
        name, min_dis = self.calculate_distance(feature)
        print('min_dis', min_dis)
        if min_dis < 0.5:
            QtWidgets.QMessageBox.information(self, '提示', f'人脸识别结果为{name}')
        else:
            QtWidgets.QMessageBox.information(self, '提示', '人脸不在数据库中')
        #self.face_img.setVisible(False)


    def register(self):
        # 弹窗提示 先打开摄像头
        if self.faces is None:
            QApplication.setQuitOnLastWindowClosed(False)
            QtWidgets.QMessageBox.warning(self, '提示', '请先打开摄像头并确保画面中有人脸')
        else:
            face = np.ascontiguousarray(
                self.frame[self.faces[0][1]:self.faces[0][3], self.faces[0][0]:self.faces[0][2]])
            #self.face_img.set_image(QImage(face.data, face.shape[1], face.shape[0], face.shape[1] * 3,
                                           #QImage.Format_RGB888))
            #self.face_img.setVisible(True)
            reply1 = QtWidgets.QMessageBox.question(self, '提示', '是否要使用该图像进行特征提取', QMessageBox.No | QMessageBox.Yes)
            if reply1 == QMessageBox.StandardButton.Yes:
                feature = self.fs.get_face_feature(Image.fromarray(face))
                if self.features.shape[0] != 0:
                    name, min_dis = self.calculate_distance(feature)
                    print(min_dis)
                    if min_dis < 0.5:
                        QtWidgets.QMessageBox.warning(self, '提示', f'该人脸已经注册过!')
                    else:
                        name, ok = QInputDialog.getText(self, '输入', '请输入唯一代号', QLineEdit.Normal, '唯一代号')
                        if ok:
                            self.save_image(face, '../images/train/' + str(name) + '/' + 'face_0.jpg')
                            self.save_image(face, '../images/train/' + str(name) + '/' + 'face_1.jpg')
                            self.train()
                            self.fs = FaceSystem()
                            feature = self.fs.get_face_feature(Image.fromarray(face))
                            self.save_feature(feature, str(name))
                            QtWidgets.QMessageBox.information(self, '提示', f'人脸注册成功！')
                else:
                    name, ok = QInputDialog.getText(self, '输入', '请输入唯一代号', QLineEdit.Normal, '唯一代号')
                    if ok:
                        self.save_image(face, '../images/train/' + str(name) + '/' + 'face_0.jpg')
                        self.save_image(face, '../images/train/' + str(name) + '/' + 'face_1.jpg')
                        self.train()
                        self.fs = FaceSystem()
                        feature = self.fs.get_face_feature(Image.fromarray(face))
                        self.save_feature(feature, str(name))
                        QtWidgets.QMessageBox.information(self, '提示', f'人脸注册成功！')
            #self.face_img.setVisible(False)

            # name, ok = QInputDialog.getText(self, '输入', '请输入姓名')
            # if ok:
            #     face_img.close()

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
                # success表示是否读取到帧， frame是读取到的numpy格式图片
                success, frame = self.cap.read()
                # 读取到video组件的长宽
                w, h = self.main.video.width(), self.main.video.height()
                # 将numpy格式进行转换成PIL格式进行向前传播
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
                    faces = faces.astype(np.int)
                    # 将人脸坐标画出来
                    frame = draw_bboxes(frame, faces)
                    # 将faces在内存空间连续排列
                    self.faces = np.ascontiguousarray(faces)
                self.frame = np.ascontiguousarray(np.array(frame))
                # 将PIL格式转为QImage self.frame.data返回存储数据指针，在使用前确保连续存储
                img = QImage(self.frame.data, self.frame.shape[1], self.frame.shape[0], self.frame.shape[1] * 3,
                             QImage.Format_RGB888)
                if self.open_flag:
                    # 将图片显示到QLabel标签中其中main.vedio是一个标签
                    self.main.video.setPixmap(QPixmap.fromImage(img))
                cv2.waitKey(int(1000 / 30))

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

    def train(self):
        img_path = '../images/train/'
        num_epochs = 10
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        transform = transforms.Compose([
            transforms.Resize((140, 140)),
            # transforms.RandomHorizontalFlip(),  # 随机水平翻转
            # transforms.RandomRotation(10),  # 随机旋转
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.6071, 0.4609, 0.3944],
                std=[0.2457, 0.2175, 0.2129]
            )
        ])

        # 加载自定义数据集
        dataset = CustomFaceDataset(root_dir=img_path, transform=transform)
        dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

        # 初始化预训练模型
        if os.path.exists('../face/facenet/weights/transferred_facenet_model.pt'):
            model = Resnet34Triplet(embedding_dimension=512)
            checkpoint = torch.load('../face/facenet/weights/transferred_facenet_model.pt')
            model.load_state_dict(checkpoint)
        else:
            checkpoint = torch.load('../face/facenet/weights/model_resnet34_triplet.pt', map_location=device)
            model = Resnet34Triplet(embedding_dimension=checkpoint['embedding_dimension'])
            model.load_state_dict(checkpoint['model_state_dict'])

        # 定义损失函数和优化器
        criterion = nn.TripletMarginLoss(margin=0.5)
        optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
        scheduler = StepLR(optimizer, step_size=5, gamma=0.1)
        # 创建窗口进度条
        progress_dialog = QProgressDialog("Training in progress...", "Cancel", 0, num_epochs, self)
        progress_dialog.setWindowTitle("Training Progress")
        progress_dialog.setWindowModality(Qt.WindowModal)

        model.to(device)
        for epoch in range(num_epochs):
            train_step(model, dataloader, criterion, optimizer, epoch, num_epochs)
            scheduler.step()
            progress_dialog.setValue(epoch + 1)
            if progress_dialog.wasCanceled():
                break

if __name__ == '__main__':
    app = QApplication([])
    window = FaceWindow()
    app.exec_()